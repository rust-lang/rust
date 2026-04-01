// A simple smoke test: when rustc compiles with profiling enabled, a profraw file
// should be generated.
// See https://github.com/rust-lang/rust/pull/48346

//@ needs-profiler-runtime
// Reason: this exercises LTO profiling
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{cwd, has_extension, has_prefix, run, rustc, shallow_find_files};

fn main() {
    rustc().opt_level("3").arg("-Clto=fat").profile_generate(cwd()).input("test.rs").run();
    run("test");
    assert_eq!(
        shallow_find_files(cwd(), |path| {
            has_prefix(path, "default_") && has_extension(path, "profraw")
        })
        .len(),
        1
    );
}
