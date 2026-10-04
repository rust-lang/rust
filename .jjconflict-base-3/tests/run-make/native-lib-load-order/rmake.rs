// An old compiler bug from 2015 caused native libraries to be loaded in the
// wrong order, causing `b` to be loaded before `a` in this test. If the compilation
// is successful, the libraries were loaded in the correct order.

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("a");
    build_native_static_lib("b");
    rustc().input("a.rs").run();
    rustc().input("b.rs").run();
    run("b");
}
