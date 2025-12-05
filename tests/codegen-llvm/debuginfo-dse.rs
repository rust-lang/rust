//@ compile-flags: -Copt-level=3 -g -Zverify-llvm-ir -Zmerge-functions=disabled
//@ revisions: CODEGEN OPTIMIZED
//@[CODEGEN] compile-flags: -Cno-prepopulate-passes
//@ only-64bit
// ignore-tidy-linelength

#![crate_type = "lib"]
#![feature(repr_simd, rustc_attrs)]

// The pass mode is direct and the backend represent is scalar.
type Scalar = i32; // scalar(i32)
type Scalar_Ref = &'static i32; // scalar(ptr)

// The pass modes are pair and the backend represents are scalar pair.
type Tuple_Scalar_Scalar = (i32, i32);
struct Tuple_Ref_Scalar(&'static i32, i32);
struct Tuple_ArrayRef_Scalar(&'static [i32; 16], i32); // pair(ptr, i32)
impl Default for Tuple_ArrayRef_Scalar {
    fn default() -> Tuple_ArrayRef_Scalar {
        Tuple_ArrayRef_Scalar(&[0; 16], 0)
    }
}
struct Tuple_Scalar_ArrayRef(i32, &'static [i32; 16]); // pair(i32, ptr)
impl Default for Tuple_Scalar_ArrayRef {
    fn default() -> Tuple_Scalar_ArrayRef {
        Tuple_Scalar_ArrayRef(0, &[0; 16])
    }
}
// The pass mode is indirect and the backend represent is memory.
type Tuple_SliceRef_Scalar = (&'static [i32], i32);

// The pass mode is pair and the backend represent is scalar pair.
type SliceRef = &'static [i32]; // pair(ptr, i32)
// The pass mode is indirect and the backend represent is memory.
type Array = [i32; 16];
// The pass mode is direct and the backend represent is scalar.
type ArrayRef = &'static [i32; 16];

// The pass mode is indirect and the backend represent is memory.
type Typle_i32_i64_i8 = (i32, i64, i8);
// The pass mode is indirect and the backend represent is memory.
#[repr(C)]
struct Aggregate_i32_Array_i8(i32, &'static [i32; 16], i8);

type ZST = ();

impl Default for Aggregate_i32_Array_i8 {
    fn default() -> Aggregate_i32_Array_i8 {
        Aggregate_i32_Array_i8(0, &[0; 16], 0)
    }
}
// The pass mode is cast and the backend represent is scalar.
#[derive(Default)]
struct Aggregate_4xi8(i8, i8, i8, i8); // scalar(i32)

// The pass mode is indirect and the backend represent is simd vector.
#[repr(simd)]
struct Simd_i32x4([i32; 4]);

unsafe extern "Rust" {
    #[rustc_nounwind]
    safe fn opaque_fn();
    #[rustc_nounwind]
    safe fn opaque_ptr(_: *const core::ffi::c_void);
}

#[inline(never)]
#[rustc_nounwind]
fn opaque_use<T>(p: &T) {
    opaque_ptr(&raw const p as *const _);
}

#[inline(never)]
#[rustc_nounwind]
fn opaque_read<T: Default>() -> T {
    core::hint::black_box(T::default())
}

#[unsafe(no_mangle)]
fn local_var() {
    // CHECK-LABEL: define{{( dso_local)?}} void @local_var
    let local_var_scalar: Scalar = opaque_read();
    opaque_use(&local_var_scalar);
    let dead_local_var_scalar: Scalar = opaque_read();
    let local_var_aggregate_4xi8: Aggregate_4xi8 = opaque_read();
    opaque_use(&local_var_aggregate_4xi8);
    let local_var_aggregate_i32_array_i8: Aggregate_i32_Array_i8 = opaque_read();
    opaque_use(&local_var_aggregate_i32_array_i8);
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // CHECK-NEXT: #dbg_value(ptr %local_var_scalar, [[ref_local_var_scalar:![0-9]+]], !DIExpression()
    let ref_local_var_scalar = &local_var_scalar;
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_dead_local_var_scalar:![0-9]+]], !DIExpression()
    let ref_dead_local_var_scalar = &dead_local_var_scalar;
    // CHECK-NEXT: #dbg_value(ptr %local_var_aggregate_4xi8, [[ref_local_var_aggregate_4xi8:![0-9]+]], !DIExpression()
    let ref_local_var_aggregate_4xi8 = &local_var_aggregate_4xi8;
    // CHECK-NEXT: #dbg_value(ptr %local_var_aggregate_4xi8, [[ref_0_local_var_aggregate_4xi8:![0-9]+]], !DIExpression()
    let ref_0_local_var_aggregate_4xi8 = &local_var_aggregate_4xi8.0;
    // CHECK-NEXT: #dbg_value(ptr %local_var_aggregate_4xi8, [[ref_2_local_var_aggregate_4xi8:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 2, DW_OP_stack_value)
    let ref_2_local_var_aggregate_4xi8 = &local_var_aggregate_4xi8.2;
    // This introduces an extra load instruction.
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_1_1_local_var_aggregate_i32_array_i8:![0-9]+]], !DIExpression()
    let ref_1_1_local_var_aggregate_i32_array_i8 = &local_var_aggregate_i32_array_i8.1[1];
    // CHECK-NEXT: #dbg_value(ptr %local_var_aggregate_i32_array_i8, [[ref_2_local_var_aggregate_i32_array_i8:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let ref_2_local_var_aggregate_i32_array_i8 = &local_var_aggregate_i32_array_i8.2;
    // CHECK: call void @opaque_fn()
    opaque_fn();
}

#[unsafe(no_mangle)]
fn zst(zst: ZST, zst_ref: &ZST) {
    // CHECK-LABEL: define{{( dso_local)?}} void @zst
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_zst:![0-9]+]], !DIExpression()
    let ref_zst = &zst;
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_zst_ref:![0-9]+]], !DIExpression()
    let ref_zst_ref = &zst_ref;
    // CHECK: call void @opaque_fn()
    opaque_fn();
}

// It only makes sense if the argument is a reference and it refer to projections.
#[unsafe(no_mangle)]
fn direct(
    scalar: Scalar,
    scalar_ref: Scalar_Ref,
    array_ref: ArrayRef,
    aggregate_4xi8_ref: &Aggregate_4xi8,
) {
    // CHECK-LABEL: define{{( dso_local)?}} void @direct
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_scalar:![0-9]+]], !DIExpression()
    let ref_scalar = &scalar;
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_scalar_ref:![0-9]+]], !DIExpression()
    let ref_scalar_ref = &scalar_ref;
    // CHECK-NEXT: #dbg_value(ptr %array_ref, [[ref_0_array_ref:![0-9]+]], !DIExpression()
    let ref_0_array_ref = &array_ref[0];
    // CHECK-NEXT: #dbg_value(ptr %array_ref, [[ref_1_array_ref:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)
    let ref_1_array_ref = &array_ref[1];
    // CHECK-NEXT: #dbg_value(ptr %aggregate_4xi8_ref, [[ref_1_aggregate_4xi8_ref:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)
    let ref_1_aggregate_4xi8_ref = &aggregate_4xi8_ref.1;
    // CHECK: call void @opaque_fn()
    opaque_fn();
}

// Arguments are passed through registers, the final values are poison.
#[unsafe(no_mangle)]
fn cast(aggregate_4xi8: Aggregate_4xi8) {
    // CHECK-LABEL: define{{( dso_local)?}} void @cast(i32 %0)
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // The temporary allocated variable is eliminated.
    // CODEGEN-NEXT: #dbg_value(ptr %aggregate_4xi8, [[ref_aggregate_4xi8:![0-9]+]], !DIExpression()
    // OPTIMIZED-NEXT: #dbg_value(ptr undef, [[ref_aggregate_4xi8:![0-9]+]], !DIExpression()
    let ref_aggregate_4xi8 = &aggregate_4xi8;
    // CODEGEN-NEXT: #dbg_value(ptr %aggregate_4xi8, [[ref_0_aggregate_4xi8:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)
    // OPTIMIZED-NEXT: #dbg_value(ptr undef, [[ref_0_aggregate_4xi8:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)
    let ref_0_aggregate_4xi8 = &aggregate_4xi8.1;
    // CHECK: call void @opaque_fn()
    opaque_fn();
}

// Arguments are passed indirectly via a pointer.
// The reference of argument is the pointer itself.
#[unsafe(no_mangle)]
fn indirect(
    tuple_sliceref_scalar: Tuple_SliceRef_Scalar,
    array: Array,
    typle_i32_i64_i8: Typle_i32_i64_i8,
    simd_i32x4: Simd_i32x4,
) {
    // CHECK-LABEL: define{{( dso_local)?}} void @indirect
    // CHECK-SAME: (ptr{{.*}} %tuple_sliceref_scalar, ptr{{.*}} %array, ptr{{.*}} %typle_i32_i64_i8, ptr{{.*}} %simd_i32x4)
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // CHECK-NEXT: #dbg_value(ptr %tuple_sliceref_scalar, [[ref_tuple_sliceref_scalar:![0-9]+]], !DIExpression()
    let ref_tuple_sliceref_scalar = &tuple_sliceref_scalar;
    // CHECK-NEXT: #dbg_value(ptr %tuple_sliceref_scalar, [[ref_1_tuple_sliceref_scalar:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let ref_1_tuple_sliceref_scalar = &tuple_sliceref_scalar.1;
    // CHECK-NEXT: #dbg_value(ptr %array, [[ref_1_array:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)
    let ref_1_array = &array[1];
    // CHECK-NEXT: #dbg_value(ptr %typle_i32_i64_i8, [[ref_1_typle_i32_i64_i8:![0-9]+]], !DIExpression()
    let ref_1_typle_i32_i64_i8 = &typle_i32_i64_i8.1;
    // CHECK-NEXT: #dbg_value(ptr %simd_i32x4, [[ref_simd_i32x4:![0-9]+]], !DIExpression()
    let ref_simd_i32x4 = &simd_i32x4;
    // CHECK: call void @opaque_fn()
    opaque_fn();
}

// They are different MIR statements, but they have the same LLVM IR statement due to the ABI of arguments.
// Both `direct_ref` and `indirect_byval` are passed as a pointer here.
#[unsafe(no_mangle)]
fn direct_ref_and_indirect(
    direct_ref: &Aggregate_i32_Array_i8,
    indirect_byval: Aggregate_i32_Array_i8,
) {
    // CHECK-LABEL: define{{( dso_local)?}} void @direct_ref_and_indirect
    // CHECK-SAME: (ptr{{.*}} %direct_ref, ptr{{.*}} %indirect_byval)
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_direct_ref:![0-9]+]], !DIExpression()
    let ref_direct_ref: &&Aggregate_i32_Array_i8 = &direct_ref;
    // CHECK-NEXT: #dbg_value(ptr %direct_ref, [[ref_1_direct_ref:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value)
    let ref_1_direct_ref = &direct_ref.1;
    // CHECK-NEXT: #dbg_value(ptr %indirect_byval, [[ref_indirect_byval:![0-9]+]], !DIExpression()
    let ref_indirect_byval: &Aggregate_i32_Array_i8 = &indirect_byval;
    // CHECK-NEXT: #dbg_value(ptr %indirect_byval, [[ref_1_indirect_byval:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value)
    let ref_1_indirect_byval = &indirect_byval.1;
    // CHECK: call void @opaque_fn()
    opaque_fn();
}

#[unsafe(no_mangle)]
fn pair(
    tuple_scalar_scalar: Tuple_Scalar_Scalar,
    tuple_ref_scalar: Tuple_Ref_Scalar,
    tuple_arrayref_scalar: Tuple_ArrayRef_Scalar,
    tuple_scalar_arrayref: Tuple_Scalar_ArrayRef,
    sliceref: SliceRef,
) {
    // CHECK-LABEL: define{{( dso_local)?}} void @pair
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_0_tuple_scalar_scalar:![0-9]+]], !DIExpression()
    let ref_0_tuple_scalar_scalar = &tuple_scalar_scalar.0;
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_0_tuple_ref_scalar:![0-9]+]], !DIExpression()
    let ref_0_tuple_ref_scalar = &tuple_ref_scalar.0;
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_1_tuple_ref_scalar:![0-9]+]], !DIExpression()
    let ref_1_tuple_ref_scalar = &tuple_ref_scalar.1;
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_0_tuple_arrayref_scalar:![0-9]+]], !DIExpression()
    let ref_0_tuple_arrayref_scalar = &tuple_arrayref_scalar.0;
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_1_tuple_arrayref_scalar:![0-9]+]], !DIExpression()
    let ref_1_tuple_arrayref_scalar = &tuple_arrayref_scalar.1;
    // FIXME: This can be a valid value.
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_0_1_tuple_arrayref_scalar:![0-9]+]], !DIExpression()
    let ref_0_1_tuple_arrayref_scalar = &tuple_arrayref_scalar.0[1];
    // FIXME: This can be a valid value.
    // CHECK-NEXT: #dbg_value(ptr poison, [[ref_1_1_tuple_scalar_arrayref:![0-9]+]], !DIExpression()
    let ref_1_1_tuple_scalar_arrayref = &tuple_scalar_arrayref.1[1];
    // CHECK: #dbg_value(ptr %sliceref.0, [[ref_1_sliceref:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)
    let ref_1_sliceref = &sliceref[1];
    // CHECK: call void @opaque_fn()
    opaque_fn();
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Foo(i32, i64, i32);

#[repr(C)]
pub struct Bar<'a> {
    a: i32,
    b: i64,
    foo: &'a Foo,
}

#[unsafe(no_mangle)]
pub fn dead_first(dead_first_foo: &Foo) -> &i32 {
    // CHECK-LABEL: def {{.*}} ptr @dead_first
    // CHECK-SAME: (ptr {{.*}} [[ARG_dead_first_foo:%.*]])
    // CODEGEN: #dbg_declare(ptr %dead_first_foo.dbg.spill, [[ARG_dead_first_foo:![0-9]+]], !DIExpression()
    // OPTIMIZED: #dbg_value(ptr %dead_first_foo, [[ARG_dead_first_foo:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr %dead_first_foo, [[VAR_dead_first_v0:![0-9]+]], !DIExpression()
    // CHECK: %dead_first_v0 = getelementptr{{.*}} i8, ptr %dead_first_foo, i64 16
    // CODEGEN: #dbg_declare(ptr %dead_first_v0.dbg.spill, [[VAR_dead_first_v0]], !DIExpression()
    // OPTIMIZED: #dbg_value(ptr %dead_first_v0, [[VAR_dead_first_v0]], !DIExpression()
    let mut dead_first_v0 = &dead_first_foo.0;
    dead_first_v0 = &dead_first_foo.2;
    dead_first_v0
}

#[unsafe(no_mangle)]
pub fn fragment(fragment_v1: Foo, mut fragment_v2: Foo) -> Foo {
    // CHECK-LABEL: define{{( dso_local)?}} void @fragment
    // CHECK-SAME: (ptr {{.*}}, ptr {{.*}} [[ARG_fragment_v1:%.*]], ptr {{.*}} [[ARG_fragment_v2:%.*]])
    // CHECK: #dbg_declare(ptr [[ARG_fragment_v1]]
    // CHECK-NEXT: #dbg_declare(ptr [[ARG_fragment_v2]]
    // CHECK-NEXT: #dbg_value(ptr [[ARG_fragment_v2]], [[VAR_fragment_f:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
    // CHECK-NEXT: #dbg_value(ptr [[ARG_fragment_v1]], [[VAR_fragment_f:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
    let fragment_f = || {
        fragment_v2 = fragment_v1;
    };
    fragment_v2 = fragment_v1;
    fragment_v2
}

#[unsafe(no_mangle)]
pub fn deref(bar: Bar) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @deref
    // We are unable to represent dereference within this expression.
    // CHECK: #dbg_value(ptr poison, [[VAR_deref_dead:![0-9]+]], !DIExpression()
    let deref_dead = &bar.foo.2;
    bar.a
}

#[unsafe(no_mangle)]
fn index(slice: &[i32; 4], idx: usize) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @index
    // CHECK: call void @opaque_fn()
    opaque_fn();
    // CHECK: #dbg_value(ptr poison, [[VAR_index_from_var:![0-9]+]], !DIExpression()
    let index_from_var = &slice[idx];
    // CHECK: #dbg_value(ptr %slice, [[VAR_const_index_from_start:![0-9]+]], !DIExpression()
    // CHECK-NEXT: #dbg_value(ptr poison, [[VAR_const_index_from_end:![0-9]+]], !DIExpression()
    let [ref const_index_from_start, .., ref const_index_from_end] = slice[..] else {
        return 0;
    };
    slice[0]
}

// CHECK-DAG: [[ref_local_var_scalar]] = !DILocalVariable(name: "ref_local_var_scalar"
// CHECK-DAG: [[ref_dead_local_var_scalar]] = !DILocalVariable(name: "ref_dead_local_var_scalar"
// CHECK-DAG: [[ref_local_var_aggregate_4xi8]] = !DILocalVariable(name: "ref_local_var_aggregate_4xi8"
// CHECK-DAG: [[ref_0_local_var_aggregate_4xi8]] = !DILocalVariable(name: "ref_0_local_var_aggregate_4xi8"
// CHECK-DAG: [[ref_2_local_var_aggregate_4xi8]] = !DILocalVariable(name: "ref_2_local_var_aggregate_4xi8"
// CHECK-DAG: [[ref_1_1_local_var_aggregate_i32_array_i8]] = !DILocalVariable(name: "ref_1_1_local_var_aggregate_i32_array_i8"
// CHECK-DAG: [[ref_2_local_var_aggregate_i32_array_i8]] = !DILocalVariable(name: "ref_2_local_var_aggregate_i32_array_i8"

// CHECK-DAG: [[ref_zst]] = !DILocalVariable(name: "ref_zst"
// CHECK-DAG: [[ref_zst_ref]] = !DILocalVariable(name: "ref_zst_ref"

// CHECK-DAG: [[ref_scalar]] = !DILocalVariable(name: "ref_scalar"
// CHECK-DAG: [[ref_scalar_ref]] = !DILocalVariable(name: "ref_scalar_ref"
// CHECK-DAG: [[ref_0_array_ref]] = !DILocalVariable(name: "ref_0_array_ref"
// CHECK-DAG: [[ref_1_array_ref]] = !DILocalVariable(name: "ref_1_array_ref"
// CHECK-DAG: [[ref_1_aggregate_4xi8_ref]] = !DILocalVariable(name: "ref_1_aggregate_4xi8_ref"

// CHECK-DAG: [[ref_aggregate_4xi8]] = !DILocalVariable(name: "ref_aggregate_4xi8"
// CHECK-DAG: [[ref_0_aggregate_4xi8]] = !DILocalVariable(name: "ref_0_aggregate_4xi8"

// CHECK-DAG: [[ref_tuple_sliceref_scalar]] = !DILocalVariable(name: "ref_tuple_sliceref_scalar"
// CHECK-DAG: [[ref_1_tuple_sliceref_scalar]] = !DILocalVariable(name: "ref_1_tuple_sliceref_scalar"
// CHECK-DAG: [[ref_1_array]] = !DILocalVariable(name: "ref_1_array"
// CHECK-DAG: [[ref_1_typle_i32_i64_i8]] = !DILocalVariable(name: "ref_1_typle_i32_i64_i8"
// CHECK-DAG: [[ref_simd_i32x4]] = !DILocalVariable(name: "ref_simd_i32x4"

// CHECK-DAG: [[ref_direct_ref]] = !DILocalVariable(name: "ref_direct_ref"
// CHECK-DAG: [[ref_1_direct_ref]] = !DILocalVariable(name: "ref_1_direct_ref"
// CHECK-DAG: [[ref_indirect_byval]] = !DILocalVariable(name: "ref_indirect_byval"
// CHECK-DAG: [[ref_1_indirect_byval]] = !DILocalVariable(name: "ref_1_indirect_byval"

// CHECK-DAG: [[ref_0_tuple_scalar_scalar]] = !DILocalVariable(name: "ref_0_tuple_scalar_scalar"
// CHECK-DAG: [[ref_0_tuple_ref_scalar]] = !DILocalVariable(name: "ref_0_tuple_ref_scalar"
// CHECK-DAG: [[ref_1_tuple_ref_scalar]] = !DILocalVariable(name: "ref_1_tuple_ref_scalar"
// CHECK-DAG: [[ref_0_tuple_arrayref_scalar]] = !DILocalVariable(name: "ref_0_tuple_arrayref_scalar"
// CHECK-DAG: [[ref_1_tuple_arrayref_scalar]] = !DILocalVariable(name: "ref_1_tuple_arrayref_scalar"
// CHECK-DAG: [[ref_0_1_tuple_arrayref_scalar]] = !DILocalVariable(name: "ref_0_1_tuple_arrayref_scalar"
// CHECK-DAG: [[ref_1_1_tuple_scalar_arrayref]] = !DILocalVariable(name: "ref_1_1_tuple_scalar_arrayref"
// CHECK-DAG: [[ref_1_sliceref]] = !DILocalVariable(name: "ref_1_sliceref"

// CHECK-DAG: [[ARG_dead_first_foo]] = !DILocalVariable(name: "dead_first_foo"
// CHECK-DAG: [[VAR_dead_first_v0]] = !DILocalVariable(name: "dead_first_v0"

// CHECK-DAG: [[VAR_fragment_f]] = !DILocalVariable(name: "fragment_f"

// CHECK-DAG: [[VAR_deref_dead]] = !DILocalVariable(name: "deref_dead"

// CHECK-DAG: [[VAR_index_from_var]] = !DILocalVariable(name: "index_from_var"
// CHECK-DAG: [[VAR_const_index_from_start]] = !DILocalVariable(name: "const_index_from_start"
// CHECK-DAG: [[VAR_const_index_from_end]] = !DILocalVariable(name: "const_index_from_end"
