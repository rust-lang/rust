// The order in which "library search path" `-L` arguments are given to the command line rustc
// is important. These arguments must match the order of the linker's arguments. In this test,
// fetching the Wrong library before the Correct one causes a function to return 0 instead of the
// expected 1, causing a runtime panic, as expected.
// See https://github.com/rust-lang/rust/pull/16904

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, path, rfs, run, run_fail, rustc, static_lib_name};

fn main() {
    build_native_static_lib("correct");
    build_native_static_lib("wrong");
    rfs::create_dir("correct");
    rfs::create_dir("wrong");
    rfs::rename(static_lib_name("correct"), path("correct").join(static_lib_name("foo")));
    rfs::rename(static_lib_name("wrong"), path("wrong").join(static_lib_name("foo")));
    rustc()
        .input("main.rs")
        .output("should_succeed")
        .library_search_path("correct")
        .library_search_path("wrong")
        .run();
    run("should_succeed");
    rustc()
        .input("main.rs")
        .output("should_fail")
        .library_search_path("wrong")
        .library_search_path("correct")
        .run();
    run_fail("should_fail");
}
