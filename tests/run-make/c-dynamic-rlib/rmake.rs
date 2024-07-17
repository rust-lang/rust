// This test checks that dynamic Rust linking with C does not encounter any errors in both
// compilation and execution, with static dependencies given preference over dynamic.
// (This is the default behaviour.)
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile
// Reason: the compiled binary is executed

//FIXME(Oneirical): test on apple because older versions of osx are failing apparently

use run_make_support::{
    build_native_dynamic_lib, dynamic_lib_name, fs_wrapper, run, run_fail, rustc,
};

fn main() {
    build_native_dynamic_lib("cfoo");
    rustc().input("foo.rs").run();
    rustc().input("bar.rs").run();
    run("bar");
    fs_wrapper::remove_file(dynamic_lib_name("cfoo"));
    run_fail("bar");
}
