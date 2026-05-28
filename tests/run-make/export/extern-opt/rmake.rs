//@ ignore-cross-compile

// NOTE: `sdylib`'s platform support is basically that of `dylib`.
//@ needs-crate-type: dylib

use run_make_support::{dynamic_lib_name, rustc};

fn main() {
    rustc().env("RUSTC_FORCE_RUSTC_VERSION", "1").input("libr.rs").run();

    rustc()
        .env("RUSTC_FORCE_RUSTC_VERSION", "2")
        .input("app.rs")
        .extern_("libr", "libinterface.rs")
        .extern_("libr", dynamic_lib_name("libr"))
        .run();

    rustc()
        .env("RUSTC_FORCE_RUSTC_VERSION", "2")
        .input("app.rs")
        .extern_("libr", "interface.rs") // wrong interface format
        .extern_("libr", dynamic_lib_name("libr"))
        .run_fail()
        .assert_stderr_contains("extern location for libr does not exist");
}
