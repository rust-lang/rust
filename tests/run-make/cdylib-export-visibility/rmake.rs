// This test builds `foo.rs` into a `cdylib` and verifies that
// `#[export_visibility = ...]` affects visibility of symbols.
//
// This test is loosely based on manual test steps described when
// discussing the related RFC at:
// https://github.com/rust-lang/rfcs/pull/3834#issuecomment-3403039933

use run_make_support::{dynamic_lib_name, llvm_readobj, rustc};

fn main() {
    // Compile a cdylib
    rustc().input("foo.rs").arg("-Zdefault-visibility=hidden").run();
    let out =
        llvm_readobj().arg("--dyn-symbols").input(dynamic_lib_name("foo")).run().stdout_utf8();

    // `#[no_mangle]` with no other attributes means: publicly exported function.
    assert!(out.contains("test_fn_no_attr"), "{out}");

    // `#[no_mangle]` with `#[export_visibility = "target_default"]` means
    // that visibility is inherited from `-Zdefault-visibility=hidden`.
    assert!(!out.contains("test_fn_export_visibility_asks_for_target_default"), "{out}");
}
