// -C profile-generate, when used with rustc, is supposed to output
// profile files (.profraw) after running a binary to analyze how the compiler
// optimizes code. This test checks that these files are generated.
// See https://github.com/rust-lang/rust/pull/48346

//@ needs-profiler-support
//@ ignore-cross-compile

use run_make_support::{cwd, find_files, has_extension, has_prefix, run, rustc};

fn main() {
    rustc().arg("-g").profile_generate(cwd()).input("test.rs").run();
    run("test");
    assert!(
        !find_files(cwd(), |path| {
            has_prefix(path, "default") && has_extension(path, "profraw")
        })
        .is_empty(),
        "no .profraw file generated"
    );
}
