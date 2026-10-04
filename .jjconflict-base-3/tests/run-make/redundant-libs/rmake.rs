// rustc will remove one of the two redundant references to foo below.  Depending
// on which one gets removed, we'll get a linker error on SOME platforms (like
// Linux).  On these platforms, when a library is referenced, the linker will
// only pull in the symbols needed _at that point in time_.  If a later library
// depends on additional symbols from the library, they will not have been pulled
// in, and you'll get undefined symbols errors.
//
// So in this example, we need to ensure that rustc keeps the _later_ reference
// to foo, and not the former one.

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_dynamic_lib, build_native_static_lib, run, rustc};

fn main() {
    build_native_dynamic_lib("foo");
    build_native_static_lib("bar");
    build_native_static_lib("baz");
    rustc()
        .args(&["-lstatic=bar", "-lfoo", "-lstatic=baz", "-lfoo"])
        .input("main.rs")
        .print("link-args")
        .run();
    run("main");
}
