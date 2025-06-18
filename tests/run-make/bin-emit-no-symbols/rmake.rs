//@ needs-target-std
//
// When setting the crate type as a "bin" (in app.rs),
// this could cause a bug where some symbols would not be
// emitted in the object files. This has been fixed, and
// this test checks that the correct symbols have been successfully
// emitted inside the object files.
// See https://github.com/rust-lang/rust/issues/51671

use run_make_support::{llvm_readobj, rustc};

fn main() {
    rustc().emit("obj").input("app.rs").run();
    let out = llvm_readobj().input("app.o").arg("--symbols").run();
    out.assert_stdout_contains("rust_begin_unwind");
    out.assert_stdout_contains("rust_eh_personality");
    out.assert_stdout_contains("__rg_oom");
}
