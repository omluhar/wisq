# resynth.py
from bqskit import compile, Circuit, MachineModel
from bqskit.ir.gates import (
    CXGate,
    RZGate,
    HGate,
    XGate,
    RXGate,
    RYGate,
    RXXGate,
    U1Gate,
    U2Gate,
    U3Gate,
    SXGate,
)
from bqskit.compiler import Compiler
from bqskit.ext import bqskit_to_qiskit
from bqskit.ext import qiskit_to_bqskit
from qiskit import QuantumCircuit
from qiskit import qasm2
import argparse
import urllib.parse
import socketserver
from http.server import BaseHTTPRequestHandler
import warnings
import time
import subprocess
import os
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary as sel
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
)
import shutil
import numpy as np
from qiskit.quantum_info import Operator
import json
import platform
from functools import partial
import sys

# Additional imports for splitting/parallelizing
from concurrent.futures import ThreadPoolExecutor
from qiskit.converters import dag_to_circuit, circuit_to_dag

LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")

# begin code from https://github.com/eth-sri/synthetiq/blob/main/notebooks/post_processing/analyzer.py
NON_STANDARD_GATES = {
    "scz": (
        "crz(pi)",
        Operator(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])),
    ),
    "U": (
        "crx(pi)",
        Operator(
            np.array(
                [
                    [
                        -0.35355339 + 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 - 0.35355339j,
                    ],
                    [
                        0.35355339 - 0.35355339j,
                        0.35355339 + 0.35355339j,
                        -0.35355339 - 0.35355339j,
                        0.35355339 - 0.35355339j,
                    ],
                    [
                        0.35355339 - 0.35355339j,
                        -0.35355339 - 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 - 0.35355339j,
                    ],
                    [
                        0.35355339 - 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 + 0.35355339j,
                        -0.35355339 + 0.35355339j,
                    ],
                ]
            )
        ),
    ),
}


class Circuit:
    def __init__(self, filename) -> None:
        with open(filename, "r") as file:
            qasm_str = file.read()
        for replace_gate in NON_STANDARD_GATES:
            qasm_str = qasm_str.replace(
                replace_gate, NON_STANDARD_GATES[replace_gate][0]
            )
        self.circuit = QuantumCircuit.from_qasm_str(qasm_str)
        for index, gate in enumerate(self.circuit.data):
            for replace_gate in NON_STANDARD_GATES:
                if (
                    gate[0].name == NON_STANDARD_GATES[replace_gate][0].split("(")[0]
                    and gate[0].params
                    and len(gate[0].params) > 0
                    and gate[0].params[0] == np.pi
                ):
                    # set gate to the correct gate operator (for nonstandard gates)
                    self.circuit.data[index] = (
                        NON_STANDARD_GATES[replace_gate][1],
                        gate[1],
                        gate[2],
                    )

        self.filename = filename
        # filename expected to contain score/count in original repo; keep compatibility
        try:
            self.score = float(os.path.basename(filename).split("-")[0])
        except Exception:
            self.score = 0.0
        self.t_depth = self.circuit.depth(lambda gate: gate[0].name in ["t", "tdg"])
        self.cx_depth = self.circuit.depth(lambda gate: gate[0].name == "cx")
        self.cx_count = np.count_nonzero(
            np.array([el[0].name for el in self.circuit.data]) == "cx"
        )
        gates_names = np.array([el[0].name for el in self.circuit.data])
        self.t_count = np.count_nonzero(gates_names == "t") + np.count_nonzero(
            gates_names == "tdg"
        )
        try:
            self.count = float(os.path.basename(filename).split("-")[1])
        except Exception:
            self.count = 0.0
        self.gates = len(gates_names)


def main_analysis(circuit_folder):
    t_depth = []
    t_count = []
    gates = []
    best_t_depth_circ = None
    best_t_count_circ = None
    best_cx_depth_circ = None
    for file in os.listdir(circuit_folder):
        circuit = Circuit(os.path.join(circuit_folder, file))
        t_depth.append(circuit.t_depth)
        gates.append(circuit.gates)
        t_count.append(circuit.t_count)

        if best_t_count_circ is None:
            best_t_count_circ = circuit
        condition2 = circuit.t_count < best_t_count_circ.t_count
        condition3 = (
            circuit.t_count == best_t_count_circ.t_count
            and circuit.t_depth < best_t_count_circ.t_depth
        )
        condition4 = (
            circuit.t_count == best_t_count_circ.t_count
            and circuit.t_depth == best_t_count_circ.t_depth
            and circuit.score < best_t_count_circ.score
        )
        if condition2 or condition3 or condition4:
            best_t_count_circ = circuit

    t_depth = np.array(t_depth)
    t_count = np.array(t_count)
    gates = np.array(gates)
    return (
        t_depth,
        t_count,
        gates,
        best_t_count_circ,
        best_t_depth_circ,
        best_cx_depth_circ,
    )


# end analyzer

warnings.filterwarnings("ignore")

GATE_SET_DICT = {
    "ibm_new": ["cx", "rz", "sx", "x"],
    "nam": ["cx", "rz", "h", "x"],
    "ion": {RXXGate(), RZGate(), RXGate(), RYGate()},
}


# ------------------------------
# New: Splitting / Parallel optimizing / Combining pipeline
# ------------------------------
MAX_BQSKIT_QUBITS = 7  # BQSKit max

def split_circuit_by_qubits(qc: QuantumCircuit, target_chunk_size: int = 30):
    if len(qc.data) == 0:
        return []

    total_gates = len(qc.data)
    subcircuits = []
    start = 0

    while start < total_gates:
        block_size = target_chunk_size
        gates_block = qc.data[start:start + block_size]

        # find qubits used in this block
        qubits_in_block = []
        for instr, qargs, _ in gates_block:
            for q in qargs:
                if q not in qubits_in_block:
                    qubits_in_block.append(q)
        # split if too many qubits
        if len(qubits_in_block) > MAX_BQSKIT_QUBITS:
            qubits_in_block = qubits_in_block[:MAX_BQSKIT_QUBITS]

        sub_qc = QuantumCircuit(len(qubits_in_block))
        qubit_map = {q: idx for idx, q in enumerate(qubits_in_block)}

        # add only gates that use qubits within the allowed set
        for instr, qargs, cargs in gates_block:
            if all(q in qubit_map for q in qargs):
                sub_qc.append(instr, [qubit_map[q] for q in qargs], cargs)

        subcircuits.append((sub_qc, qubits_in_block))
        start += block_size

    print(f"[Split] Total gates={total_gates}. Created {len(subcircuits)} subcircuits.")
    return subcircuits


def optimize_subcircuits_parallel(
    compiler,
    subcircuits,
    opt_level: int = 2,
    epsilon: float = 1e-6,
    target_gateset: str = "ibm_new",
    max_workers: int | None = None,
):
    """
    Optimizes subcircuits in parallel.
    Each item in `subcircuits` is a tuple (QuantumCircuit, qubit_list)
    Returns list of (optimized_subcircuit, qubit_list)
    """
    if not subcircuits:
        return []

    def do_opt(item):
        sub_qc, qubits_in_block = item
        qasm_str = qasm2.dumps(sub_qc)
        optimized_qasm = bqskit_io(compiler, {}, qasm_str, opt_level, epsilon, target_gateset)
        optimized_qc = QuantumCircuit.from_qasm_str(optimized_qasm)
        return (optimized_qc, qubits_in_block)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(do_opt, subcircuits))

    print(f"[Optimize] Optimized {len(results)} subcircuits (parallel).")
    return results



def combine_subcircuits(subcircuits_with_qubits, total_qubits):
    """
    Combine subcircuits into a single circuit with `total_qubits` qubits.
    subcircuits_with_qubits: list of (QuantumCircuit, qubit_list)
    """
    combined = QuantumCircuit(total_qubits)
    for sub, qubits_in_block in subcircuits_with_qubits:
        # map back to original qubits
        combined.compose(sub, qubits=qubits_in_block, inplace=True)
    print(f"[Combine] Recombined circuit has {combined.size()} total gates.")
    return combined


# ------------------------------
# Test helper that checks equivalence (unitary) between original and combined
# ------------------------------
def test_parallel_optimizer(
    compiler=None,
    target_gateset="ibm_new",
    target_chunk_size=15,  # smaller chunk size ensures multiple subcircuits
    num_qubits=5,
    num_layers=20          # reduce layers to keep unitary small enough for BQSKit
):
    """
    Generate a test circuit, split -> parallel-optimize -> combine,
    and check unitary equivalence.
    """
    print(f"[Test] Building test circuit with {num_qubits} qubits and {num_layers} layers.")
    qc = QuantumCircuit(num_qubits)

    # Fixed, repeatable pattern of gates
    for _ in range(num_layers):
        for q in range(num_qubits):
            qc.h(q)
            qc.rz(np.pi / 4, q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    print(f"[Test] Generated circuit with {len(qc.data)} gates.")

    # Split circuit into subcircuits
    subs = split_circuit_by_qubits(qc, target_chunk_size=target_chunk_size)
    print(f"[Test] Created {len(subs)} subcircuits (each <= {target_chunk_size} gates).")

    if len(subs) < 2:
        print("[Warning] Only one subcircuit was created. Consider reducing target_chunk_size.")

    # Optimize subcircuits in parallel
    optimized_subs = optimize_subcircuits_parallel(
        compiler, subs, opt_level=2, epsilon=1e-8, target_gateset=target_gateset
    )

    # Combine optimized subcircuits
    combined = combine_subcircuits(optimized_subs, total_qubits=num_qubits)

    # Check unitary equivalence
    orig_op = Operator(qc).data
    combined_op = Operator(combined).data
    max_diff = np.max(np.abs(orig_op - combined_op))
    equivalent = np.allclose(orig_op, combined_op, atol=1e-6)

    print(f"[Test] Max unitary difference: {max_diff:.3e}")
    print("[Test] Equivalent?" , "YES" if equivalent else "NO")

    return combined


# ------------------------------
# Existing bqskit_io, synthetiq_disk, server code unchanged (except we now call the new pipeline)
# ------------------------------
def bqskit_io(compiler, data, circuit_str, opt_level, epsilon, target_gateset):
    qc = QuantumCircuit.from_qasm_str(circuit_str)
    data["circuit"] = circuit_str
    data["original_size"] = qc.size()
    data["original_2q_size"] = qc.num_nonlocal_gates()
    model = (
        MachineModel(qc.num_qubits, gate_set=GATE_SET_DICT[target_gateset])
        if target_gateset == "ion"
        else None
    )
    circuit = compile(
        qiskit_to_bqskit(qc).get_unitary(),
        optimization_level=opt_level,
        synthesis_epsilon=epsilon,
        compiler=compiler,
        model=model,
    )
    data["bqskit_params"] = {"opt_level": opt_level, "epsilon": epsilon}
    data["resynth_size"] = circuit.num_operations
    data["resynth_2q_size"] = (
        circuit.gate_counts[CXGate()] if CXGate() in circuit.gate_counts else 0
    ) + (circuit.gate_counts[RXXGate()] if RXXGate() in circuit.gate_counts else 0)

    if target_gateset != "none" and target_gateset != "ion":
        circuit = bqskit_to_qiskit(circuit)
        pm = PassManager(
            [
                BasisTranslator(sel, GATE_SET_DICT[target_gateset]),
            ]
        )
        circuit = pm.run(circuit)
        return qasm2.dumps(circuit)

    return circuit.to("qasm")


def get_t_count(circuit):
    count_ops = circuit.count_ops()
    return count_ops.get("t", 0) + count_ops.get("tdg", 0)


def synthetiq_disk(
    data,
    circuit_str,
    num_circuits,
    epsilon,
    threads,
    target_gateset,
    verbose,
    path_to_synthetiq,
):
    qc = QuantumCircuit.from_qasm_str(circuit_str)
    matrix = Operator(qc).data
    data["circuit"] = circuit_str
    data["original_size"] = qc.size()
    data["original_t_size"] = get_t_count(qc)
    data["original_2q_size"] = qc.num_nonlocal_gates()

    temp_circ = f"circ_{np.random.randint(0, 1000000)}"

    with open(f"{LIB_DIR}/synthetiq/data/input/{temp_circ}.txt", "w") as f:
        f.write(f"{temp_circ}\n")
        f.write(f"{qc.num_qubits}\n")
        for row in matrix:
            for val in row:
                f.write(f"({val.real},{val.imag}) ")
            f.write("\n")
        for row in matrix:
            for val in row:
                f.write(f"1 ")
            f.write("\n")

    directory = f"{LIB_DIR}/synthetiq/data/output/{temp_circ}"
    temp_circ_path = f"{LIB_DIR}/synthetiq/data/input/{temp_circ}.txt"

    command = f"{path_to_synthetiq} {temp_circ}.txt -c {num_circuits} -eps {epsilon} -h {threads}"
    data["synthetiq_command"] = command
    command_list = command.split(" ")
    proc = subprocess.Popen(
        command_list,
        cwd=os.path.join(LIB_DIR, "synthetiq"),
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.DEVNULL if not verbose else None,
    )
    proc.wait()

    (
        t_depth,
        t_count,
        gates,
        best_t_count_circ,
        best_t_depth_circ,
        best_cx_depth_circ,
    ) = main_analysis(directory)

    if os.path.exists(directory):
        shutil.rmtree(directory)
    if os.path.exists(temp_circ_path):
        os.remove(temp_circ_path)

    new_circ = best_t_count_circ.circuit
    data["resynth_size"] = new_circ.size()
    data["resynth_t_size"] = get_t_count(new_circ)
    data["resynth_2q_size"] = new_circ.num_nonlocal_gates()

    if target_gateset != "none":
        pm = PassManager(
            [
                BasisTranslator(sel, GATE_SET_DICT[target_gateset]),
            ]
        )
        new_circ = pm.run(new_circ)

    # rename qreg because synthetiq uses "qubits" by default
    return qasm2.dumps(new_circ).replace("qubits[", qc.qregs[0].name + "[")


class MyHandler(BaseHTTPRequestHandler):
    def __init__(
        self, bqskit, bqskit_auto_workers, verbose, path_to_synthetiq, *args, **kwargs
    ):
        self.compiler = None
        if bqskit:
            if bqskit_auto_workers:
                self.compiler = Compiler()
            else:
                self.compiler = Compiler(num_workers=64)
        self.verbose = verbose
        self.path_to_synthetiq = path_to_synthetiq
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/bqskit":
            parsed_body = json.loads(body)
            time1 = time.time()
            data = {}
            output = bqskit_io(
                self.compiler,
                data,
                parsed_body["circuit"],
                int(parsed_body["opt_level"]),
                float(parsed_body["epsilon"]),
                parsed_body["target_gateset"],
            )
            data["resynthesized_circuit"] = output
            time2 = time.time()
            data["time"] = time2 - time1
            print(data)
        if parsed_path.path == "/synthetiq":
            parsed_body = json.loads(body)
            time1 = time.time()
            data = {}
            output = synthetiq_disk(
                data,
                parsed_body["circuit"],
                int(parsed_body["num_circuits"]),
                float(parsed_body["epsilon"]),
                int(parsed_body["threads"]),
                parsed_body["target_gateset"],
                self.verbose,
                self.path_to_synthetiq,
            )
            data["resynthesized_circuit"] = output
            time2 = time.time()
            data["time"] = time2 - time1
            print(data)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(output.encode("utf-8"))

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()


def start_server(bqskit, bqskit_auto_workers, verbose=False, path_to_synthetiq=None):
    if not verbose:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    partial_handler = partial(
        MyHandler, bqskit, bqskit_auto_workers, verbose, path_to_synthetiq
    )
    try:
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", 8080), partial_handler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "--bqskit",
        action=argparse.BooleanOptionalAction,
        help="Use BQSKit (initialize BQSKit compiler instance). Not necessary if using some other resynthesis algorithm (e.g. Synthetiq).",
    )
    parser.add_argument(
        "--bqskit_auto_workers",
        action=argparse.BooleanOptionalAction,
        help="[Recommended] Use BQSKit default mechanism for determining how many workers to spin up",
    )
    parser.add_argument(
        "--path_to_synthetiq",
        type=str,
        help="Absolute path to Synthetiq `main` binary",
        default=os.path.join(LIB_DIR, "synthetiq", "bin", "main"),
    )

    args = parser.parse_args()

    # Initialize Compiler if requested by server mode
    if args.bqskit:
        # If user passed --bqskit, we will create a Compiler for server endpoints to use.
        # Note: test below will also create a Compiler for the parallel test.
        server_compiler = Compiler()
    else:
        server_compiler = None

    # Run the parallel optimizer test (uses Compiler if available)
    print("[Main] Running parallel optimizer test (small circuit) ...")
    # create a Compiler for the test run regardless (keeps behavior similar to original script)
    try:
        test_compiler = Compiler()
    except Exception:
        test_compiler = None
    combined_qc = test_parallel_optimizer(
        test_compiler,
        target_gateset="ibm_new",
        target_chunk_size=30,
        num_qubits=5,   # Adjust as needed
        num_layers=50   # Adjust layers to make circuit bigger
    )
    print("[Main] Test finished. Combined circuit size:", combined_qc.size())

    # Start server (same behavior as original file)
    start_server(
        args.bqskit,
        args.bqskit_auto_workers,
        verbose=True,
        path_to_synthetiq=args.path_to_synthetiq,
    )