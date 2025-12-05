// Exercise unwinding a panic. This catches a panic across an FFI (foreign function interface)
// boundary and downcasts it into an integer.
// The Rust code that panics is in a separate crate.
// See https://github.com/rust-lang/rust/commit/baf227ea0c1e07fc54395a51e4b3881d701180cb

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ needs-unwind
// Reason: this test exercises unwinding a panic

use run_make_support::{cc, is_windows_msvc, llvm_ar, run, rustc, static_lib_name};

fn main() {
    // Compile `add.c` into an object file.
    if is_windows_msvc() {
        cc().arg("-c").out_exe("add").input("add.c").run();
    } else {
        cc().arg("-v").arg("-c").out_exe("add.o").input("add.c").run();
    };

    // Compile `panic.rs` into an object file.
    // Note that we invoke `rustc` directly, so we may emit an object rather
    // than an archive. We'll do that later.
    rustc().emit("obj").input("panic.rs").run();

    // Now, create an archive using these two objects.
    if is_windows_msvc() {
        llvm_ar().obj_to_ar().args(&[&static_lib_name("add"), "add.obj", "panic.o"]).run();
    } else {
        llvm_ar().obj_to_ar().args(&[&static_lib_name("add"), "add.o", "panic.o"]).run();
    };

    // Compile `main.rs`, which will link into our library, and run it.
    rustc().input("main.rs").run();
    run("main");
}
