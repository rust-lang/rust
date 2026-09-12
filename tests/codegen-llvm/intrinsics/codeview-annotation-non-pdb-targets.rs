// Verifies that `codeview_annotation` does NOT emit `llvm.codeview.annotation`
// on targets not using PDB debuginfo which is everything except MSVC and UEFI.

//@ ignore-msvc
//@ ignore-uefi
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(codeview_annotation)]
use std::hint::{CodeViewAnnotationArgs, codeview_annotation};

struct Args;

impl CodeViewAnnotationArgs for Args {
    const ARGS: &[&str] = &["string1", "string2", "string3"];
}

// CHECK-LABEL: @test_non_pdb
// CHECK-NOT: llvm.codeview.annotation
#[no_mangle]
pub fn test_non_pdb() {
    codeview_annotation::<Args>();
}
