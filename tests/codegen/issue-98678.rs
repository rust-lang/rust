// This test verifies the accuracy of emitted file and line debuginfo metadata.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]
#![feature(generators, stmt_expr_attributes)]

// The use of CHECK-DAG here is because the C++-like enum is emitted before the `DIFile` node

// CHECK-DAG: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}src/test/codegen/issue-98678.rs{{".*}})

// CHECK-DAG: !DICompositeType({{.*"}}MyCppLikeEnum{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
#[repr(C)]
pub enum MyCppLikeEnum {
    One,
}

// CHECK: !DICompositeType({{.*"}}MyType{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub struct MyType {
    // CHECK: !DIDerivedType({{.*"}}i{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    i: i32,
}

// CHECK: !DICompositeType({{.*"}}MyUnion{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub union MyUnion {
    // CHECK: !DIDerivedType({{.*"}}i{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    i: i32,
    // CHECK: !DIDerivedType({{.*"}}f{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    f: f32,
}

// CHECK: !DICompositeType({{.*"}}MyNativeEnum{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub enum MyNativeEnum {
    One,
}

pub fn foo(_: MyType, _: MyUnion, _: MyNativeEnum, _: MyCppLikeEnum) {
    // CHECK: !DICompositeType({{.*"[{]}}closure_env#0{{[}]".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let closure = |x| x;
    closure(0);

    // CHECK: !DICompositeType({{.*"[{]}}generator_env#1{{[}]".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let generator = #[coroutine]
    || yield 1;
}
