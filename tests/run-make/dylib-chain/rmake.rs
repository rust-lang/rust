// In this test, m4 depends on m3, which depends on m2, which depends on m1.
// Even though dependencies are chained like this and there is no direct mention
// of m1 or m2 in m4.rs, compilation and execution should still succeed. Unlike the
// rlib-chain test, dynamic libraries contain upstream dependencies, and breaking
// the chain by removing the dylibs causes execution to fail.
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{dynamic_lib_name, rfs, run, run_fail, rustc};

fn main() {
    rustc().input("m1.rs").arg("-Cprefer-dynamic").run();
    rustc().input("m2.rs").arg("-Cprefer-dynamic").run();
    rustc().input("m3.rs").arg("-Cprefer-dynamic").run();
    rustc().input("m4.rs").run();
    run("m4");
    rfs::remove_file(dynamic_lib_name("m1"));
    rfs::remove_file(dynamic_lib_name("m2"));
    rfs::remove_file(dynamic_lib_name("m3"));
    run_fail("m4");
}
