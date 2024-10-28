//! Check that linking frameworks on Apple platforms works.

//@ only-apple

use run_make_support::{Rustc, run, rustc};

fn compile(cfg: &str) -> Rustc {
    let mut rustc = rustc();
    rustc.cfg(cfg).input("main.rs");
    rustc
}

fn main() {
    for cfg in ["link", "weak", "both"] {
        compile(cfg).run();
        run("main");
    }

    let errs = compile("omit").run_fail();
    // The linker's exact error output changes between Xcode versions, depends on
    // linker invocation details, and the linker sometimes outputs more warnings.
    errs.assert_stderr_contains_regex(r"error: linking with `.*` failed");
    errs.assert_stderr_contains_regex(r"(Undefined symbols|ld: symbol[^\s]* not found)");
    errs.assert_stderr_contains_regex(r".?_CFRunLoopGetTypeID.?, referenced from:");
    errs.assert_stderr_contains("clang: error: linker command failed with exit code 1");
}
