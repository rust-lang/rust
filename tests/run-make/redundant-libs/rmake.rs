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
//@ ignore-windows-msvc
// Reason: this test links libraries via link.exe, which only accepts the import library
// for the dynamic library, i.e. `foo.dll.lib`. However, build_native_dynamic_lib only
// produces `foo.dll` - the dynamic library itself. To make this test work on MSVC, one
// would need to derive the import library from the dynamic library.
// See https://stackoverflow.com/questions/9360280/

use run_make_support::{
    build_native_dynamic_lib, build_native_static_lib, cwd, is_msvc, rfs, run, rustc,
};

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
