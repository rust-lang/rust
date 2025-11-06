//@ compile-flags: -Z annotate-moves=8 -Copt-level=0 -g
//
// This test verifies that function call and return instructions use the correct debug scopes
// when passing/returning large values. The actual move/copy operations may be annotated,
// but the CALL and RETURN instructions themselves should reference the source location,
// NOT have an inlinedAt scope pointing to compiler_move/compiler_copy.

#![crate_type = "lib"]

#[derive(Clone, Copy)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

#[derive(Clone, Copy)]
pub struct MediumStruct {
    pub data: [u64; 5], // 40 bytes
}

pub struct SmallStruct {
    pub x: u32, // 4 bytes
}

// ============================================================================
// Test 1: Single argument call
// ============================================================================

// CHECK-LABEL: call_arg_scope::test_call_with_single_arg
pub fn test_call_with_single_arg(s: LargeStruct) {
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#CALL1_ARG_LOC:]]
    // CHECK: call {{.*}}@{{.*}}helper_single{{.*}}({{.*}}){{.*}}, !dbg ![[#CALL1_LOC:]]
    helper_single(s);
}

#[inline(never)]
fn helper_single(_s: LargeStruct) {}

// ============================================================================
// Test 2: Multiple arguments of different types
// ============================================================================

// CHECK-LABEL: call_arg_scope::test_call_with_multiple_args
pub fn test_call_with_multiple_args(large: LargeStruct, medium: MediumStruct, small: SmallStruct) {
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#CALL2_ARG1_LOC:]]
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#CALL2_ARG2_LOC:]]
    // CHECK: call {{.*}}@{{.*}}helper_multiple{{.*}}({{.*}}){{.*}}, !dbg ![[#CALL2_LOC:]]
    helper_multiple(large, medium, small);
}

#[inline(never)]
fn helper_multiple(_l: LargeStruct, _m: MediumStruct, _s: SmallStruct) {}

// ============================================================================
// Test 3: Return value
// ============================================================================

// CHECK-LABEL: call_arg_scope::test_return_large_value
pub fn test_return_large_value() -> LargeStruct {
    let s = LargeStruct { data: [42; 20] };
    // CHECK: ret {{.*}}, !dbg ![[#RET1_LOC:]]
    s
}

// ============================================================================
// Test 4: Calling a function that returns a large value
// ============================================================================

// CHECK-LABEL: call_arg_scope::test_call_returning_large
pub fn test_call_returning_large() {
    // CHECK: call {{.*}}@{{.*}}make_large_struct{{.*}}({{.*}}){{.*}}, !dbg ![[#CALL3_LOC:]]
    let _result = make_large_struct();
}

#[inline(never)]
fn make_large_struct() -> LargeStruct {
    LargeStruct { data: [1; 20] }
}

// ============================================================================
// Test 5: Mixed scenario - passing and returning large values
// ============================================================================

// CHECK-LABEL: call_arg_scope::test_mixed_call
pub fn test_mixed_call(input: LargeStruct) -> LargeStruct {
    // CHECK: call {{.*}}@{{.*}}transform_large{{.*}}({{.*}}){{.*}}, !dbg ![[#CALL4_LOC:]]
    transform_large(input)
}

#[inline(never)]
fn transform_large(mut s: LargeStruct) -> LargeStruct {
    s.data[0] += 1;
    s
}

// CHECK-DAG: ![[#CALL1_ARG_LOC]] = !DILocation({{.*}}scope: ![[#CALL1_ARG_SCOPE:]]
// CHECK-DAG: ![[#CALL1_ARG_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<call_arg_scope::LargeStruct, 160>"
// CHECK-DAG: ![[#CALL1_LOC]] = !DILocation({{.*}}scope: ![[#CALL1_SCOPE:]]
// CHECK-DAG: ![[#CALL1_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "test_call_with_single_arg"

// CHECK-DAG: ![[#CALL2_ARG1_LOC]] = !DILocation({{.*}}scope: ![[#CALL2_ARG1_SCOPE:]]
// CHECK-DAG: ![[#CALL2_ARG1_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<call_arg_scope::LargeStruct, 160>"
// CHECK-DAG: ![[#CALL2_ARG2_LOC]] = !DILocation({{.*}}scope: ![[#CALL2_ARG2_SCOPE:]]
// CHECK-DAG: ![[#CALL2_ARG2_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<call_arg_scope::MediumStruct, 40>"
// CHECK-DAG: ![[#CALL2_LOC]] = !DILocation({{.*}}scope: ![[#CALL2_SCOPE:]]
// CHECK-DAG: ![[#CALL2_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "test_call_with_multiple_args"

// CHECK-DAG: ![[#CALL3_LOC]] = !DILocation({{.*}}scope: ![[#CALL3_SCOPE:]]
// CHECK-DAG: ![[#CALL3_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "test_call_returning_large"

// CHECK-DAG: ![[#CALL4_LOC]] = !DILocation({{.*}}scope: ![[#CALL4_SCOPE:]]
// CHECK-DAG: ![[#CALL4_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "test_mixed_call"

// CHECK-DAG: ![[#RET1_LOC]] = !DILocation({{.*}}scope: ![[#RET1_SCOPE:]]
// CHECK-DAG: ![[#RET1_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "test_return_large_value"
