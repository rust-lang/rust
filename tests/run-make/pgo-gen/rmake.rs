// -C profile-generate, when used with rustc, is supposed to output
// profile files (.profraw) after running a binary to analyze how the compiler
// optimizes code. This test checks that these files are generated.
// See https://github.com/rust-lang/rust/pull/48346

//@ needs-profiler-support
//@ ignore-cross-compile

use run_make_support::{cwd, find_files_by_prefix_and_extension, run, rustc};

fn main() {
    rustc().arg("-g").profile_generate(cwd()).run();
    run("test");
    assert!(
        find_files_by_prefix_and_extension(cwd(), "default", "profraw").len() > 0,
        "no .profraw file generated"
    );
}
