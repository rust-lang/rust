//@ compile-flags: -Copt-level=3 -g -Zverify-llvm-ir
//@ revisions: CODEGEN OPTIMIZED
//@[CODEGEN] compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Foo(i32, i64, i32);

#[no_mangle]
fn r#ref(ref_foo: &Foo) -> i32 {
    // CHECK-LABEL: define {{.*}} i32 @ref
    // CHECK-SAME: (ptr {{.*}} [[ARG_ref_foo:%.*]])
    // OPTIMIZED: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_foo:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_v0:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_v1:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value)
    // CHECK: #dbg_value(ptr [[ARG_ref_foo]], [[VAR_ref_v2:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let ref_v0 = &ref_foo.0;
    let ref_v1 = &ref_foo.1;
    let ref_v2 = &ref_foo.2;
    ref_foo.0
}

#[no_mangle]
fn ptr(ptr_foo: Foo) -> i32 {
    // CHECK-LABEL: define {{.*}} i32 @ptr
    // CHECK-SAME: (ptr {{.*}} [[ARG_ptr_foo:%.*]])
    // CHECK: #dbg_value(ptr [[ARG_ptr_foo]], [[VAR_ptr_v0:![0-9]+]], !DIExpression()
    // CHECK: #dbg_value(ptr [[ARG_ptr_foo]], [[VAR_ptr_v1:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value)
    // CHECK: #dbg_value(ptr [[ARG_ptr_foo]], [[VAR_ptr_v2:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let ptr_v0 = &ptr_foo.0;
    let ptr_v1 = &ptr_foo.1;
    let ptr_v2 = &ptr_foo.2;
    ptr_foo.2
}

#[no_mangle]
fn no_ptr(val: i32) -> i32 {
    // CHECK-LABEL: define {{.*}} i32 @no_ptr
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
pub fn tuple(foo: (i32, &Foo)) -> i32 {
    // CHECK-LABEL: define {{.*}} i32 @tuple
    // CHECK-SAME: (i32 {{.*}}, ptr {{.*}} [[ARG_tuple_foo_1:%.*]])
    // CHECK: #dbg_value(ptr [[ARG_tuple_foo_1]], [[VAR_tuple_dead:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_stack_value)
    let tuple_dead = &foo.1.2;
    foo.1.0
}

// OPTIMIZED-DAG: [[VAR_ref_foo]] = !DILocalVariable(name: "ref_foo"
// CHECK-DAG: [[VAR_ref_v0]] = !DILocalVariable(name: "ref_v0"
// CHECK-DAG: [[VAR_ref_v1]] = !DILocalVariable(name: "ref_v1"
// CHECK-DAG: [[VAR_ref_v2]] = !DILocalVariable(name: "ref_v2"
// CHECK-DAG: [[VAR_ptr_v0]] = !DILocalVariable(name: "ptr_v0"
// CHECK-DAG: [[VAR_ptr_v1]] = !DILocalVariable(name: "ptr_v1"
// CHECK-DAG: [[VAR_ptr_v2]] = !DILocalVariable(name: "ptr_v2"
// CODEGEN-DAG: [[VAR_val_ref]] = !DILocalVariable(name: "val_ref"
// CHECK-DAG: [[VAR_fragment_f]] = !DILocalVariable(name: "fragment_f"
// CHECK-DAG: [[VAR_tuple_dead]] = !DILocalVariable(name: "tuple_dead"
