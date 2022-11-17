// This test verifies the accuracy of emitted file and line debuginfo metadata.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]

// The use of CHECK-DAG here is because the C++-like enum is emitted before the `DIFile` node

// CHECK-DAG: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}src/test/codegen/issue-98678.rs{{".*}})

// CHECK-DAG: !DICompositeType({{.*"}}MyCppLikeEnum{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
#[repr(C)]
pub enum MyCppLikeEnum {
    One,
}

// CHECK: !DICompositeType({{.*"}}MyType{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub struct MyType;

// CHECK: !DICompositeType({{.*"}}MyUnion{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub union MyUnion {
    i: i32, // TODO fields are still wrong
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
}
