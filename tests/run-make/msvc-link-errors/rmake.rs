// Test various msvc's link.exe error scenarios.

//@ only-x86_64-pc-windows-msvc
//@ ignore-cross-compile: the built binary is executed

use run_make_support::{diff, rustc};

fn main() {
    // Ok
    rustc()
        .input("foo.rs")
        .crate_type("bin")
        .arg("-Clto=thin")
        .opt()
        .arg(&format!("-Clinker=link"))
        .run();

    // linker not found
    let out = rustc()
        .input("foo.rs")
        .crate_type("bin")
        .arg("-Clto=thin")
        .opt()
        .arg(&format!("-Clinker=link_"))
        .run_fail();
    diff()
        .expected_file("missing_link.stderr")
        .actual_text("(linker)", out.stderr())
        .run();
}
