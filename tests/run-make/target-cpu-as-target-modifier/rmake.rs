// This test verifies that only the expected built-in targets opt in to treating
// `-C target-cpu` as a target modifier.
//
// The test first asks rustc for the full list of supported built-in targets,
// then performs two checks for each target.
//
// 1. Target-spec check
//
//    The test prints the target specification as JSON and verifies that
//    `requires-consistent-cpu` is set to `true` only for the expected targets.
//    All other targets must either omit the field or set it to `false`.
//
// 2. Cross-crate compatibility check
//
//    The test builds `dependency.rs` with target CPU `A`, then builds `main.rs`
//    against that dependency under several configurations.
//
//    The `main.rs` build must succeed when:
//    - it is built with the same target CPU `A`;
//    - it is built with a different target CPU `B`, but the target does not opt
//      in to `requires-consistent-cpu`;
//    - it is built with a different target CPU `B`, the target does opt in to
//      `requires-consistent-cpu`, and the compiler is invoked with
//      `-C unsafe-allow-abi-mismatch=target-cpu`.
//
//    The `main.rs` build must fail when it is built with target CPU `B`, the
//    target opts in to `requires-consistent-cpu`, and
//    `-C unsafe-allow-abi-mismatch=target-cpu` is not used.
//
// The test only verifies the target-modifier compatibility check, so it does
// not need to run code generation and uses `--emit=metadata`.
//
// To avoid depending on whether a target is supported by the selected codegen
// backend, the test also uses `-Z codegen-backend=dummy`.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use run_make_support::*;
use serde_json::Value;

static EXPECTED: LazyLock<BTreeSet<&str>> =
    LazyLock::new(|| BTreeSet::from(["amdgcn-amd-amdhsa", "avr-none", "nvptx64-nvidia-cuda"]));

use run_make_support::rustc;
fn main() {
    verify_target_specs();
    verify_cross_crate_compatibility();
}

fn verify_target_specs() {
    let requires_consistent_cpu = |spec: &Value| {
        spec.get("requires-consistent-cpu").and_then(Value::as_bool).unwrap_or(false)
    };

    let json = rustc().arg("-Zunstable-options").print("all-target-specs-json").run().stdout_utf8();

    let specs: Value = serde_json::from_str(&json).unwrap();

    let actual = specs
        .as_object()
        .expect("expected all-target-specs-json to be a JSON object")
        .iter()
        .filter_map(|(target, spec)| requires_consistent_cpu(spec).then_some(target.as_str()))
        .collect::<BTreeSet<_>>();

    assert_eq!(
        actual, *EXPECTED,
        "unexpected set of built-in targets with `requires-consistent-cpu = true`",
    );
}

fn verify_cross_crate_compatibility() {
    let target_list = rustc().print("target-list").run().stdout_utf8();
    let targets: Vec<&str> = target_list.lines().collect();

    for target in targets.iter() {
        let compiler = |cpu: &str, input: &str| {
            let mut cmd = rustc();
            cmd.target(target)
                .target_cpu(cpu)
                .input(input)
                .panic("abort")
                .args(["--emit=metadata", "-Zcodegen-backend=dummy"]);
            cmd
        };
        let (first_cpu, second_cpu) = ("A", "B");

        // Build dependency.rs using the first target-cpu
        compiler(first_cpu, "dependency.rs").run();

        if EXPECTED.contains(target) {
            // Testing targets where `-Ctarget-cpu` acts as a target modifier:
            // Building with the same target cpu must succeed.
            compiler(first_cpu, "main.rs").run();
            // Building with a different target cpu must succeed if
            // rustc is invoked with `-Cunsafe-allow-abi-mismatch=target-cpu`
            compiler(second_cpu, "main.rs").arg("-Cunsafe-allow-abi-mismatch=target-cpu").run();
            // Building with a different target cpu must fail if
            // rustc is _not_ invoked with `-Cunsafe-allow-abi-mismatch=target-cpu`
            compiler(second_cpu, "main.rs").run_fail().assert_stderr_contains(
                "error: mixing `-Ctarget-cpu` will cause \
                     an ABI mismatch in crate `main`",
            );
        } else {
            // Testing targets where `-Ctarget-cpu` does not act as a target modifier:
            // Building with a different target cpu must succeed.
            compiler(second_cpu, "main.rs").run();
        }
    }
}
