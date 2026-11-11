// When the compiler is performing link time optimization, it will
// need to copy the original rlib file, set the copy's permissions to read/write,
// and modify that copy - even if the original
// file is read-only. This test creates a read-only rlib, and checks that
// compilation with LTO succeeds.
// See https://github.com/rust-lang/rust/pull/17619

//@ ignore-cross-compile
// FIXME: Once GCC backend is fixed, remove this `ignore-backends`.
//@ ignore-backends: gcc

use run_make_support::{run, rust_lib_name, rustc, test_while_readonly};

fn main() {
    rustc().input("lib.rs").run();
    test_while_readonly(rust_lib_name("lib"), || {
        rustc().input("main.rs").arg("-Clto").run();
        run("main");
    });
}
