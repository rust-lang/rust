// #[bench] is a Rust feature to run benchmarks on performance-critical
// code, which previously experienced a runtime panic bug in #103794.
// In order to ensure future breakages of this feature are detected, this
// smoke test was created, using the benchmarking feature with various
// runtime flags.
// See https://github.com/rust-lang/rust/issues/103794

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ needs-unwind
// Reason: #[bench] and -Zpanic-abort-tests can't be combined

use run_make_support::{run, run_with_args, rustc};

fn main() {
    // Smoke-test that #[bench] isn't entirely broken.
    rustc().arg("--test").input("smokebench.rs").opt().run();
    run_with_args("smokebench", &["--bench"]);
    run_with_args("smokebench", &["--bench", "noiter"]);
    run_with_args("smokebench", &["--bench", "yesiter"]);
    run("smokebench");
}
