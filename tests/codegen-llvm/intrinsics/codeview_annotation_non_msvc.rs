// Verifies that codeview_annotation does NOT emit `llvm.codeview.annotation`
// on non-MSVC targets. The intrinsic should be silently ignored.

//@ ignore-msvc
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(codeview_annotation)]
#![feature(core_intrinsics)]

use std::intrinsics::codeview_annotation;

// CHECK-LABEL: @test_non_msvc
// CHECK-NOT: codeview.annotation
#[no_mangle]
pub fn test_non_msvc() {
    codeview_annotation(&["string1", "string2", "string3"]);
}
