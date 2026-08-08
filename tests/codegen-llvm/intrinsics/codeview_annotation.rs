// Verifies that codeview_annotation intrinsic lowers correctly
// under various conditions like directly calling the intrinsic,
// calling through the macro, calling with different kinds of
// args and in the presence of duplicate calls.

//@ only-msvc
//@ revisions: OPT0 OPT3
//@ [OPT0] compile-flags: -Copt-level=0
//@ [OPT3] compile-flags: -Copt-level=3
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(codeview_annotation)]
#![feature(core_intrinsics)]

use std::intrinsics::codeview_annotation;

// CHECK-LABEL: @intrinsic_single_annotation
// CHECK: call void @llvm.codeview.annotation(metadata [[SINGLE:![0-9]+]])
#[no_mangle]
pub fn intrinsic_single_annotation() {
    codeview_annotation(&["intr_single_string"]);
}

// CHECK-LABEL: @intrinsic_multiple_annotations
// CHECK: call void @llvm.codeview.annotation(metadata [[MULTI:![0-9]+]])
#[no_mangle]
pub fn intrinsic_multiple_annotations() {
    codeview_annotation(&["intr_multi1", "intr_multi2", "intr_multi3"]);
}

// CHECK-LABEL: @macro_single_annotation
// CHECK: call void @llvm.codeview.annotation(metadata [[MACRO_SINGLE:![0-9]+]])
#[no_mangle]
pub fn macro_single_annotation() {
    std::hint::codeview_annotation!("mac_single");
}

// CHECK-LABEL: @macro_multiple_annotations
// CHECK: call void @llvm.codeview.annotation(metadata [[MACRO_MULTI:![0-9]+]])
#[no_mangle]
pub fn macro_multiple_annotations() {
    std::hint::codeview_annotation!("mac_multi1", "mac_multi2", "mac_multi3");
}

const STR_A: &str = "named_const1";
const STR_B: &str = "named_const2";
const STR_C: &str = "named_const3";

// CHECK-LABEL: @named_const_elements
// CHECK: call void @llvm.codeview.annotation(metadata [[NAMED_CONST:![0-9]+]])
#[no_mangle]
pub fn named_const_elements() {
    codeview_annotation(&[STR_A, STR_B, STR_C]);
}

// CHECK-LABEL: @mixed_named_const_and_literal_elements
// CHECK: call void @llvm.codeview.annotation(metadata [[MIXED:![0-9]+]])
#[no_mangle]
pub fn mixed_named_const_and_literal_elements() {
    codeview_annotation(&[STR_A, "mixed_literal1", "mixed_literal2"]);
}

const STRS_SLICE: &[&str] = &["slice_element1", "slice_element2", "slice_element3"];

// CHECK-LABEL: @named_const_slice
// CHECK: call void @llvm.codeview.annotation(metadata [[CONST_SLICE:![0-9]+]])
#[no_mangle]
pub fn named_const_slice() {
    codeview_annotation(STRS_SLICE);
}

const STRS_ARRAY: [&str; 3] = ["arr_element1", "arr_element2", "arr_element3"];

// CHECK-LABEL: @named_const_array_ref
// CHECK: call void @llvm.codeview.annotation(metadata [[CONST_ARRAY:![0-9]+]])
#[no_mangle]
pub fn named_const_array_ref() {
    codeview_annotation(&STRS_ARRAY);
}

// Multiple annotations with same strings within a single function
// CHECK-LABEL: @duplicate_annotations
// CHECK: call void @llvm.codeview.annotation(metadata [[DUP:![0-9]+]])
// CHECK: call void @llvm.codeview.annotation(metadata [[DUP]])
#[no_mangle]
pub fn duplicate_annotations() {
    codeview_annotation(&["dup1", "dup2", "dup3"]);
    codeview_annotation(&["dup1", "dup2", "dup3"]);
}

// Multiple annotations with same strings within different functions
// CHECK-LABEL: @duplicate_annotations_func_a
// CHECK: call void @llvm.codeview.annotation(metadata [[FUNC_DUP:![0-9]+]])
#[no_mangle]
pub fn duplicate_annotations_func_a() {
    codeview_annotation(&["func_dup1", "func_dup2", "func_dup3"]);
}

// CHECK-LABEL: @duplicate_annotations_func_b
// CHECK: call void @llvm.codeview.annotation(metadata [[FUNC_DUP:![0-9]+]])
#[no_mangle]
pub fn duplicate_annotations_func_b() {
    codeview_annotation(&["func_dup1", "func_dup2", "func_dup3"]);
}

// Metadata definitions are at the end of LLVM IR, so check them here
// CHECK-DAG: [[SINGLE]] = !{!"intr_single_string"}
// CHECK-DAG: [[MULTI]] = !{!"intr_multi1", !"intr_multi2", !"intr_multi3"}
// CHECK-DAG: [[MACRO_SINGLE]] = !{!"mac_single"}
// CHECK-DAG: [[MACRO_MULTI]] = !{!"mac_multi1", !"mac_multi2", !"mac_multi3"}
// CHECK-DAG: [[NAMED_CONST]] = !{!"named_const1", !"named_const2", !"named_const3"}
// CHECK-DAG: [[MIXED]] = !{!"named_const1", !"mixed_literal1", !"mixed_literal2"}
// CHECK-DAG: [[CONST_SLICE]] = !{!"slice_element1", !"slice_element2", !"slice_element3"}
// CHECK-DAG: [[CONST_ARRAY]] = !{!"arr_element1", !"arr_element2", !"arr_element3"}
// CHECK-DAG: [[DUP]] = !{!"dup1", !"dup2", !"dup3"}
// CHECK-DAG: [[FUNC_DUP]] = !{!"func_dup1", !"func_dup2", !"func_dup3"}
