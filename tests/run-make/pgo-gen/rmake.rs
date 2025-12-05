// -C profile-generate, when used with rustc, is supposed to output
// profile files (.profraw) after running a binary to analyze how the compiler
// optimizes code. This test checks that these files are generated.
// See https://github.com/rust-lang/rust/pull/48346

//@ needs-profiler-runtime
//@ ignore-cross-compile

use run_make_support::{cwd, has_extension, has_prefix, run, rustc, shallow_find_files};

fn main() {
    rustc().arg("-g").profile_generate(cwd()).input("test.rs").run();
    run("test");
    let profraw_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "default") && has_extension(path, "profraw")
    });
    assert!(!profraw_files.is_empty(), "no .profraw file generated");
}
