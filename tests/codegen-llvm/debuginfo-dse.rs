//@ compile-flags: -Copt-level=3 -g -Zverify-llvm-ir -Zmerge-functions=disabled
//@ revisions: CODEGEN OPTIMIZED
//@[CODEGEN] compile-flags: -Cno-prepopulate-passes
// ignore-tidy-linelength

#![crate_type = "lib"]

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Foo(i32, i64, i32);

#[repr(C)]
pub struct Bar<'a> {
    a: i32,
    b: i64,
    foo: &'a Foo,
}

#[no_mangle]
fn r#ref(ref_foo: &Foo) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @ref
    // CHECK-SAME: (ptr {{.*}} [[ARG_ref_foo:%.*]])
    // OPTIMIZED: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_foo:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr poison, [[VAR_invalid_ref_of_ref_foo:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_v0:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_v1:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value)
    // CHECK: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_v2:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let invalid_ref_of_ref_foo = &ref_foo;
    let ref_v0 = &ref_foo.0;
    let ref_v1 = &ref_foo.1;
    let ref_v2 = &ref_foo.2;
    ref_foo.0
}

#[no_mangle]
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

#[no_mangle]
fn ptr(ptr_foo: Foo) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @ptr
    // CHECK-SAME: (ptr {{.*}} [[ARG_ptr_foo:%.*]])
    // CHECK: #dbg_value(ptr [[ARG_ptr_foo]], [[ref_ptr_foo:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr [[ARG_ptr_foo]], [[VAR_ptr_v0:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr [[ARG_ptr_foo]], [[VAR_ptr_v1:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value)
    // CHECK: #dbg_value(ptr [[ARG_ptr_foo]], [[VAR_ptr_v2:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let ref_ptr_foo = &ptr_foo;
    let ptr_v0 = &ptr_foo.0;
    let ptr_v1 = &ptr_foo.1;
    let ptr_v2 = &ptr_foo.2;
    ptr_foo.2
}

#[no_mangle]
fn no_ptr(val: i32) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @no_ptr
    // CODEGEN: #dbg_value(ptr poison, [[VAR_val_ref:![0-9]+]], !DIExpression()
    let val_ref = &val;
    val
}

#[no_mangle]
pub fn fragment(fragment_v1: Foo, mut fragment_v2: Foo) -> Foo {
    // CHECK-LABEL: define void @fragment
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

#[no_mangle]
pub fn deref(bar: Bar) -> i32 {
    // CHECK-LABEL: define {{.*}} i32 @deref
    // We are unable to represent dereference within this expression.
    // CHECK: #dbg_value(ptr poison, [[VAR_deref_dead:![0-9]+]], !DIExpression()
    let deref_dead = &bar.foo.2;
    bar.a
}

#[no_mangle]
pub fn tuple(foo: (i32, &Foo)) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @tuple
    // Although there is no dereference here, there is a dereference in the MIR.
    // CHECK: #dbg_value(ptr poison, [[VAR_tuple_dead:![0-9]+]], !DIExpression()
    let tuple_dead = &foo.1.2;
    foo.1.0
}

pub struct ZST;

#[no_mangle]
pub fn zst(zst: ZST, v: &i32) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @zst
    // CHECK: #dbg_value(ptr poison, [[VAR_zst_ref:![0-9]+]], !DIExpression()
    let zst_ref = &zst;
    *v
}

#[no_mangle]
fn index(slice: &[i32; 4], idx: usize) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @index
    // CHECK: bb1:
    // CHECK-NEXT: #dbg_value(ptr poison, [[VAR_index_from_var:![0-9]+]], !DIExpression()
    // CODEGEN: bb3:
    // CHECK-NEXT: #dbg_value(ptr %slice, [[VAR_const_index_from_start:![0-9]+]], !DIExpression()
    // CHECK-NEXT: #dbg_value(ptr poison, [[VAR_const_index_from_end:![0-9]+]], !DIExpression()
    let index_from_var = &slice[idx];
    let [ref const_index_from_start, .., ref const_index_from_end] = slice[..] else {
        return 0;
    };
    slice[0]
}

unsafe extern "Rust" {
    safe fn opaque_inner(_: *const core::ffi::c_void);
}

#[inline(never)]
pub fn opaque_use<T>(p: &T) {
    opaque_inner(&raw const p as *const _);
}

#[no_mangle]
pub fn non_arg_ref(scalar: i32, foo: Foo, a: &i32) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @non_arg_ref
    // CHECK: #dbg_value(ptr %non_arg_ref_scalar, [[VAR_non_arg_ref_scalar_ref:![0-9]+]], !DIExpression()
    // CHECK-NEXT: #dbg_value(ptr %non_arg_ref_foo, [[VAR_non_arg_ref_foo_ref:![0-9]+]], !DIExpression()
    // CHECK-NEXT: #dbg_value(ptr %non_arg_ref_foo, [[VAR_non_arg_ref_foo_ref_2:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let non_arg_ref_scalar = scalar;
    let non_arg_ref_foo = foo;
    opaque_use(&non_arg_ref_scalar);
    opaque_use(&non_arg_ref_foo);
    let non_arg_ref_scalar_ref = &non_arg_ref_scalar;
    let non_arg_ref_foo_ref = &non_arg_ref_foo;
    let non_arg_ref_foo_ref_2 = &non_arg_ref_foo.2;
    *a
}

// CHECK-DAG: [[VAR_invalid_ref_of_ref_foo]] = !DILocalVariable(name: "invalid_ref_of_ref_foo"
// OPTIMIZED-DAG: [[VAR_ref_foo]] = !DILocalVariable(name: "ref_foo"
// CHECK-DAG: [[VAR_ref_v0]] = !DILocalVariable(name: "ref_v0"
// CHECK-DAG: [[VAR_ref_v1]] = !DILocalVariable(name: "ref_v1"
// CHECK-DAG: [[VAR_ref_v2]] = !DILocalVariable(name: "ref_v2"
// CHECK-DAG: [[ref_ptr_foo]] = !DILocalVariable(name: "ref_ptr_foo"
// CHECK-DAG: [[VAR_ptr_v0]] = !DILocalVariable(name: "ptr_v0"
// CHECK-DAG: [[VAR_ptr_v1]] = !DILocalVariable(name: "ptr_v1"
// CHECK-DAG: [[VAR_ptr_v2]] = !DILocalVariable(name: "ptr_v2"
// CODEGEN-DAG: [[VAR_val_ref]] = !DILocalVariable(name: "val_ref"
// CHECK-DAG: [[VAR_fragment_f]] = !DILocalVariable(name: "fragment_f"
// CHECK-DAG: [[VAR_deref_dead]] = !DILocalVariable(name: "deref_dead"
// CHECK-DAG: [[VAR_tuple_dead]] = !DILocalVariable(name: "tuple_dead"
// CHECK-DAG: [[ARG_dead_first_foo]] = !DILocalVariable(name: "dead_first_foo"
// CHECK-DAG: [[VAR_dead_first_v0]] = !DILocalVariable(name: "dead_first_v0"
// CHECK-DAG: [[VAR_index_from_var]] = !DILocalVariable(name: "index_from_var"
// CHECK-DAG: [[VAR_const_index_from_start]] = !DILocalVariable(name: "const_index_from_start"
// CHECK-DAG: [[VAR_const_index_from_end]] = !DILocalVariable(name: "const_index_from_end"
// CHECK-DAG: [[VAR_zst_ref]] = !DILocalVariable(name: "zst_ref"
// CHECK-DAG: [[VAR_non_arg_ref_scalar_ref]] = !DILocalVariable(name: "non_arg_ref_scalar_ref"
// CHECK-DAG: [[VAR_non_arg_ref_foo_ref]] = !DILocalVariable(name: "non_arg_ref_foo_ref"
// CHECK-DAG: [[VAR_non_arg_ref_foo_ref_2]] = !DILocalVariable(name: "non_arg_ref_foo_ref_2"
