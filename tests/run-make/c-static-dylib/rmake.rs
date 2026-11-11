// This test checks that static Rust linking with C does not encounter any errors,
// with dynamic dependencies given preference over static.
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{
    build_native_static_lib, dynamic_lib_name, rfs, run, run_fail, rustc, static_lib_name,
};

fn main() {
    build_native_static_lib("cfoo");
    rustc().input("foo.rs").arg("-Cprefer-dynamic").run();
    rustc().input("bar.rs").run();
    rfs::remove_file(static_lib_name("cfoo"));
    run("bar");
    rfs::remove_file(dynamic_lib_name("foo"));
    run_fail("bar");
}
