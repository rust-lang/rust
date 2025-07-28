//! Regression test for edge-case bugs in coverage instrumentation that have
//! historically been triggered by derived `arbitrary::Arbitrary` impls.
//!
//! See <https://github.com/rust-lang/rust/issues/141577#issuecomment-3120667286>
//! for an example of one such bug.

//@ needs-profiler-runtime

use run_make_support::{cargo, is_windows, llvm};

fn main() {
    let profraw_path = "default.profraw";
    let profdata_path = "default.profdata";

    // Build and run the crate with coverage instrumentation,
    // producing a `.profraw` file.
    let run_out = cargo()
        .args(&["run", "--manifest-path=Cargo.toml", "--release"])
        .env("RUSTFLAGS", "-Cinstrument-coverage")
        .env("LLVM_PROFILE_FILE", profraw_path)
        .run();

    // The program prints its own executable path (i.e. args[0]) to stdout.
    let exe_path = run_out.stdout_utf8().lines().next().unwrap().to_owned();

    // Convert `.profraw` output to `.profdata`, as needed by `llvm-cov`.
    llvm::llvm_profdata()
        .args(&["merge", "--sparse", "--output", profdata_path, profraw_path])
        .run();

    // The contents of the coverage report are not very important;
    // what matters is that `llvm-cov` should not encounter an error
    // (e.g. "malformed instrumentation profile data: function name is empty").
    llvm::llvm_cov()
        .args(&["show", "-format=text", "-instr-profile", profdata_path, "-object", &exe_path])
        .run();
}
