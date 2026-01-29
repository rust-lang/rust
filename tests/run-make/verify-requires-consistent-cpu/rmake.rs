//! Verifies that only for the targets mentioned in `must_fail` `-Ctarget-cpu` acts as a
//! target-modifier and for all others it does not.

//@ ignore-backends: gcc

use std::collections::BTreeSet;
use std::panic;

use run_make_support::{llvm_components_contain, rustc, util};

fn main() {
    let must_fail = BTreeSet::from(["avr-none", "amdgcn-amd-amdhsa", "nvptx64-nvidia-cuda"]);
    let target_list = rustc().print("target-list").run().stdout_utf8();
    let targets: Vec<&str> = target_list.lines().collect();

    // FIXME: We need to filter xtensa because of a data-layout mismatch
    for target in targets.iter().filter(|&str| !str.contains("xtensa")) {
        // FIXME: riscv targets list incompatible CPUs, so we hard-code the used
        // target-cpus for them
        let (first_target_cpu, second_target_cpu) = if target.contains("riscv32") {
            if !llvm_components_contain("riscv") {
                continue;
            }
            ("generic-rv32".to_string(), "rocket-rv32".to_string())
        } else if target.contains("riscv64") {
            if !llvm_components_contain("riscv") {
                continue;
            }
            ("generic-rv64".to_string(), "rocket-rv64".to_string())
        } else {
            let mut cmd = rustc();
            let completed = cmd.target(target).print("target-cpus").run_unchecked();
            // We guard against missing LLVM components
            if completed
                .stderr_utf8()
                .contains("error: could not create LLVM TargetMachine for triple")
            {
                continue;
            } else if !completed.status().success() {
                util::handle_failed_output(
                    cmd.as_ref(),
                    completed,
                    panic::Location::caller().line(),
                );
            }
            let output = completed.stdout_utf8();
            // Take the first two cpus which are not "native"
            let mut cpus = output
                .lines()
                .skip(1)
                .filter_map(|str| str.split_whitespace().next())
                .filter(|&str| str != "native")
                .take(2);
            (
                cpus.next()
                    .expect(format!("First target cpu was not found for {target}").as_str())
                    .to_string(),
                cpus.next()
                    .expect(format!("Second target cpu was not found for {target}").as_str())
                    .to_string(),
            )
        };

        // Build depedency.rs the first target-cpu
        let mut dep_cmd = rustc();
        dep_cmd.target(target).target_cpu(&first_target_cpu).input("dependency.rs").run();

        if must_fail.contains(target) {
            // Test targets where `-Ctarget-cpu` acts as a target modifier.
            let mut allow_match = rustc();
            allow_match
                .target(target)
                .target_cpu(&first_target_cpu)
                .input("incompatible_target_cpu.rs")
                .panic("abort")
                .run();
            let mut allow_mismatch = rustc();
            allow_mismatch
                .target(target)
                .target_cpu(&second_target_cpu)
                .arg("-Cunsafe-allow-abi-mismatch=target-cpu")
                .input("incompatible_target_cpu.rs")
                .panic("abort")
                .run();
            let mut error_generated = rustc();
            error_generated
                .target(target)
                .target_cpu(&second_target_cpu)
                .input("incompatible_target_cpu.rs")
                .panic("abort")
                .run_fail()
                .assert_stderr_contains(
                    "error: mixing `-Ctarget-cpu` will cause \
                     an ABI mismatch in crate `incompatible_target_cpu`",
                );
        } else {
            // Test targets where `-Ctarget-cpu` does not act as a target modifier
            // by testing that linking different target-cpus does not lead to an error.
            let mut cmd = rustc();
            cmd.target(target)
                .target_cpu(&second_target_cpu)
                .input("incompatible_target_cpu.rs")
                .panic("abort")
                .run();
        }
    }
}
