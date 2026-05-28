// This test checks that dynamic Rust linking with C does not encounter any errors in both
// compilation and execution, with dynamic dependencies given preference over static.
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_dynamic_lib, dynamic_lib_name, rfs, run, run_fail, rustc};

fn main() {
    build_native_dynamic_lib("cfoo");
    rustc().input("foo.rs").arg("-Cprefer-dynamic").run();
    rustc().input("bar.rs").run();
    run("bar");
    rfs::remove_file(dynamic_lib_name("cfoo"));
    run_fail("bar");
}
