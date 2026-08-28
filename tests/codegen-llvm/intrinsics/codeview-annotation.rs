// Verifies that codeview_annotation lowers correctly to
// `llvm.codeview.annotation`

//@ only-msvc
//@ revisions: OPT0 OPT3
//@ [OPT0] compile-flags: -Copt-level=0
//@ [OPT3] compile-flags: -Copt-level=3
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(codeview_annotation)]
#![feature(core_intrinsics)]

// === Helper macros ===
macro_rules! call_intrinsic {
    ($args:expr) => {{
        call_codeview_annotation!(std::intrinsics, $args);
    }};
}

macro_rules! call_api {
    ($args:expr) => {{
        call_codeview_annotation!(std::hint, $args);
    }};
}

macro_rules! call_codeview_annotation {
    ($($module:ident)::+, $args:expr) => {{
        struct Args;

        impl std::hint::CodeViewAnnotationArgs for Args {
            const ARGS: &[&str] = $args;
        }

        $($module)::+::codeview_annotation::<Args>();
    }};
}

// At OPT0, the API wrapper function is not inlined
// so we must check it exists and calls the intrinsic
// OPT0-LABEL: ; core::hint::codeview_annotation::<codeview_annotation::api_single_annotation::Args>
// OPT0: define internal void [[API_SINGLE_WRAPPER:@[^(]+]]()
// OPT0: call void @llvm.codeview.annotation(metadata !{{[0-9]+}})
// OPT0-LABEL: ; core::hint::codeview_annotation::<codeview_annotation::api_multiple_annotations::Args>
// OPT0: define internal void [[API_MULTIPLE_WRAPPER:@[^(]+]]()
// OPT0: call void @llvm.codeview.annotation(metadata !{{[0-9]+}})

// === Intrinsic tests ===
// CHECK-LABEL: @single
// CHECK: call void @llvm.codeview.annotation(metadata [[SINGLE:![0-9]+]])
#[no_mangle]
pub fn single() {
    call_intrinsic!(&["single_string"]);
}

// CHECK-LABEL: @multiple
// CHECK: call void @llvm.codeview.annotation(metadata [[MULTIPLE:![0-9]+]])
#[no_mangle]
pub fn multiple() {
    call_intrinsic!(&["multi1", "multi2", "multi3"]);
}

const STR_A: &str = "str_a";
const STR_B: &str = "str_b";
const STR_C: &str = "str_c";

// CHECK-LABEL: @named_const_elements
// CHECK: call void @llvm.codeview.annotation(metadata [[NAMED_CONST:![0-9]+]])
#[no_mangle]
pub fn named_const_elements() {
    call_intrinsic!(&[STR_A, STR_B, STR_C]);
}

// CHECK-LABEL: @mixed_named_consts_and_literals
// CHECK: call void @llvm.codeview.annotation(metadata [[MIXED:![0-9]+]])
#[no_mangle]
pub fn mixed_named_consts_and_literals() {
    call_intrinsic!(&[STR_A, "mixed_literal1", "mixed_literal2"]);
}

const STRS_SLICE: &[&str] = &["slice_element1", "slice_element2", "slice_element3"];

// CHECK-LABEL: @named_const_slice
// CHECK: call void @llvm.codeview.annotation(metadata [[CONST_SLICE:![0-9]+]])
#[no_mangle]
pub fn named_const_slice() {
    call_intrinsic!(STRS_SLICE);
}

const STRS_ARRAY: [&str; 3] = ["arr_element1", "arr_element2", "arr_element3"];

// CHECK-LABEL: @named_const_array_ref
// CHECK: call void @llvm.codeview.annotation(metadata [[CONST_ARRAY:![0-9]+]])
#[no_mangle]
pub fn named_const_array_ref() {
    call_intrinsic!(&STRS_ARRAY);
}

static STATIC_STRS_ARRAY: [&str; 3] = ["static1", "static2", "static3"];

// CHECK-LABEL: @static_array_ref
// CHECK: call void @llvm.codeview.annotation(metadata [[STATIC_ARRAY:![0-9]+]])
#[no_mangle]
pub fn static_array_ref() {
    call_intrinsic!(&STATIC_STRS_ARRAY);
}

static STATIC_STRING_BYTES: [u8; 5] = *b"bytes";
static STATIC_STRING: &str = unsafe { core::str::from_utf8_unchecked(&STATIC_STRING_BYTES) };

// CHECK-LABEL: @static_string_element
// CHECK: call void @llvm.codeview.annotation(metadata [[STATIC_STRING:![0-9]+]])
#[no_mangle]
pub fn static_string_element() {
    call_intrinsic!(&["string1", STATIC_STRING, "string3"]);
}

// CHECK-LABEL: @empty_strings
// CHECK: call void @llvm.codeview.annotation(metadata [[EMPTY_STRINGS:![0-9]+]])
#[no_mangle]
pub fn empty_strings() {
    call_intrinsic!(&["", "", "string1"]);
}

// CHECK-LABEL: @empty_slice
// CHECK: call void @llvm.codeview.annotation(metadata [[EMPTY_SLICE:![0-9]+]])
#[no_mangle]
pub fn empty_slice() {
    call_intrinsic!(&[]);
}

const EMPTY_STRS_SLICE: &[&str] = &[];

// CHECK-LABEL: @named_empty_slice
// CHECK: call void @llvm.codeview.annotation(metadata [[EMPTY_SLICE]])
#[no_mangle]
pub fn named_empty_slice() {
    call_intrinsic!(EMPTY_STRS_SLICE);
}

// Multiple annotations with same strings within a single function
// CHECK-LABEL: @duplicate_annotations
// CHECK: call void @llvm.codeview.annotation(metadata [[DUP:![0-9]+]])
// CHECK: call void @llvm.codeview.annotation(metadata [[DUP]])
#[no_mangle]
pub fn duplicate_annotations() {
    call_intrinsic!(&["dup1", "dup2", "dup3"]);
    call_intrinsic!(&["dup1", "dup2", "dup3"]);
}

// Multiple annotations with same strings within different functions
// CHECK-LABEL: @duplicate_annotations_func_a
// CHECK: call void @llvm.codeview.annotation(metadata [[FUNC_DUP:![0-9]+]])
#[no_mangle]
pub fn duplicate_annotations_func_a() {
    call_intrinsic!(&["func_dup1", "func_dup2", "func_dup3"]);
}

// CHECK-LABEL: @duplicate_annotations_func_b
// CHECK: call void @llvm.codeview.annotation(metadata [[FUNC_DUP:![0-9]+]])
#[no_mangle]
pub fn duplicate_annotations_func_b() {
    call_intrinsic!(&["func_dup1", "func_dup2", "func_dup3"]);
}

// === API tests ===
// CHECK-LABEL: @api_single_annotation
// OPT0: call void [[API_SINGLE_WRAPPER]]()
// OPT3: call void @llvm.codeview.annotation(metadata !{{[0-9]+}})
#[no_mangle]
pub fn api_single_annotation() {
    call_api!(&["intr_single_string"]);
}

// CHECK-LABEL: @api_multiple_annotations
// OPT0: call void [[API_MULTIPLE_WRAPPER]]()
// OPT3: call void @llvm.codeview.annotation(metadata !{{[0-9]+}})
#[no_mangle]
pub fn api_multiple_annotations() {
    call_api!(&["intr_multi1", "intr_multi2", "intr_multi3"]);
}

// Metadata definitions are at the end of LLVM IR, so check them here
// CHECK-DAG: [[SINGLE]] = !{!"single_string"}
// CHECK-DAG: [[MULTIPLE]] = !{!"multi1", !"multi2", !"multi3"}
// CHECK-DAG: [[NAMED_CONST]] = !{!"str_a", !"str_b", !"str_c"}
// CHECK-DAG: [[MIXED]] = !{!"str_a", !"mixed_literal1", !"mixed_literal2"}
// CHECK-DAG: [[CONST_SLICE]] = !{!"slice_element1", !"slice_element2", !"slice_element3"}
// CHECK-DAG: [[CONST_ARRAY]] = !{!"arr_element1", !"arr_element2", !"arr_element3"}
// CHECK-DAG: [[STATIC_ARRAY]] = !{!"static1", !"static2", !"static3"}
// CHECK-DAG: [[STATIC_STRING]] = !{!"string1", !"bytes", !"string3"}
// CHECK-DAG: [[EMPTY_STRINGS]] = !{!"", !"", !"string1"}
// CHECK-DAG: [[EMPTY_SLICE]] = !{}
// CHECK-DAG: [[DUP]] = !{!"dup1", !"dup2", !"dup3"}
// CHECK-DAG: [[FUNC_DUP]] = !{!"func_dup1", !"func_dup2", !"func_dup3"}
// CHECK-DAG: !{{[0-9]+}} = !{!"intr_single_string"}
// CHECK-DAG: !{{[0-9]+}} = !{!"intr_multi1", !"intr_multi2", !"intr_multi3"}
