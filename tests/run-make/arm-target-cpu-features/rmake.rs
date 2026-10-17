// This tests that target-cpu correctly enables additional features for some Arm targets.
// These targets were originally defined in such a way that features provided by target-cpu would be
// disabled by the target spec itself. This was fixed in #123159.

// FIXME: This test should move to tests/assembly when building without #![no_core] in
// that environment is possible, tracked by #130375.

use run_make_support::{llvm_filecheck, llvm_objdump, rustc, static_lib_name};

struct TestCase {
    target: &'static str,
    cpu: &'static str,
}

static CASES: &[TestCase] = &[
    TestCase { target: "thumbv7em-none-eabihf", cpu: "cortex-m7" },
    TestCase { target: "thumbv8m.main-none-eabihf", cpu: "cortex-m85" },
];

fn main() {
    for case in CASES {
        let lib = static_lib_name(case.cpu);
        let checks = format!("{}.checks", case.cpu);

        let rustc_command = || {
            let mut cmd = rustc();
            cmd.edition("2021")
                .target(case.target)
                .arg("-Copt-level=3")
                .crate_type("rlib")
                .input("lib.rs")
                .output(&lib);
            cmd
        };

        let objdump_command = || {
            let mut cmd = llvm_objdump();
            cmd.arg("--arch-name=arm")
                .arg(format!("--mcpu={}", case.cpu))
                .disassemble()
                .input(&lib);
            cmd
        };

        // First, run without target-cpu and confirm that it fails.
        rustc_command().run();
        let dis = objdump_command().run().stdout_utf8();
        llvm_filecheck().patterns(&checks).stdin_buf(dis).run_fail();

        // Then, run with target-cpu and confirm that it succeeds.
        rustc_command().arg(format!("-Ctarget-cpu={}", case.cpu)).run();
        let dis = objdump_command().run().stdout_utf8();
        llvm_filecheck().patterns(&checks).stdin_buf(dis).run();
    }
}
