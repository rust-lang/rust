// Symbols related to the allocator should be hidden and not exported from a cdylib,
// for they are internal to Rust
// and not part of the public ABI (application binary interface). This test checks that
// four such symbols are successfully hidden.
// See https://github.com/rust-lang/rust/pull/45710

//FIXME(Oneirical): try it on windows, restore ignore
// See https://github.com/rust-lang/rust/pull/46207#issuecomment-347561753
//FIXME(Oneirical): I also removed cross-compile ignore since there is no binary execution

use run_make_support::{dynamic_lib_name, llvm_readobj, rustc};

fn main() {
    // Compile a cdylib
    rustc().input("foo.rs").run();
    let out = llvm_readobj().arg("--symbols").input(dynamic_lib_name("foo")).run().stdout_utf8();
    let out = // All hidden symbols must be removed.
        out.lines().filter(|&line| !line.trim().contains("HIDDEN")).collect::<Vec<_>>().join("\n");
    assert!(!&out.contains("__rdl_"));
    assert!(!&out.contains("__rde_"));
    assert!(!&out.contains("__rg_"));
    assert!(!&out.contains("__ruse_"));
}
