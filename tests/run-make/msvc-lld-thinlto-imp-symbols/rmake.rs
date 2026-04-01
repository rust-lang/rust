// This is a non-regression test for issue #81408 involving an lld bug and ThinLTO, on windows.
// MSVC's link.exe doesn't need any workarounds in rustc, but lld does, so we'll check that the
// binary runs successfully instead of using a codegen test.

//@ only-x86_64-pc-windows-msvc
//@ needs-rust-lld
//@ ignore-cross-compile: the built binary is executed

use run_make_support::{run, rustc};

fn test_with_linker(linker: &str) {
    rustc().input("issue_81408.rs").crate_name("issue_81408").crate_type("lib").opt().run();
    rustc()
        .input("main.rs")
        .crate_type("bin")
        .arg("-Clto=thin")
        .opt()
        .arg(&format!("-Clinker={linker}"))
        .extern_("issue_81408", "libissue_81408.rlib")
        .run();

    // To make possible failures clearer, print an intro that will only be shown if the test does
    // fail when running the binary.
    eprint!("Running binary linked with {linker}... ");
    run("main");
    eprintln!("ok");
}

fn main() {
    // We want the reproducer to work when linked with both linkers.
    test_with_linker("link");
    test_with_linker("rust-lld");
}
