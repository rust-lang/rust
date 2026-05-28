// Symbols related to the allocator should be hidden and not exported from a cdylib,
// for they are internal to Rust
// and not part of the public ABI (application binary interface). This test checks that
// four such symbols are successfully hidden.
// See https://github.com/rust-lang/rust/pull/45710

//@ ignore-cross-compile
// Reason: The __rust_ symbol appears during cross-compilation.

use run_make_support::{dynamic_lib_name, llvm_readobj, rustc};

fn main() {
    // Compile a cdylib
    rustc().input("foo.rs").run();
    let out =
        llvm_readobj().arg("--dyn-symbols").input(dynamic_lib_name("foo")).run().stdout_utf8();
    assert!(!&out.contains("__rdl_"), "{out}");
    assert!(!&out.contains("__rde_"), "{out}");
    assert!(!&out.contains("__rg_"), "{out}");
    assert!(!&out.contains("__rust_"), "{out}");
}
