// Test that if we build `b` against a version of `a` that has
// one set of types, it will not run with a dylib that has a different set of types.

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{run, run_fail, rustc};

fn main() {
    rustc()
        .input("a.rs")
        .cfg("x")
        .arg("-Zunstable-options")
        .arg("-Cprefer-dynamic")
        .arg("-Csymbol-mangling-version=legacy")
        .run();

    rustc()
        .input("b.rs")
        .arg("-Zunstable-options")
        .arg("-Cprefer-dynamic")
        .arg("-Csymbol-mangling-version=legacy")
        .run();

    run("b");

    rustc()
        .input("a.rs")
        .cfg("y")
        .arg("-Zunstable-options")
        .arg("-Cprefer-dynamic")
        .arg("-Csymbol-mangling-version=legacy")
        .run();

    run_fail("b");
}
