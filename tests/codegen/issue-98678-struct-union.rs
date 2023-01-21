// This test verifies the accuracy of emitted file and line debuginfo metadata for structs and
// unions.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]

// NONMSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}/codegen/issue-98678-struct-union.rs{{".*}})
// MSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\codegen\\issue-98678-struct-union.rs{{".*}})

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

pub fn foo(_: MyType, _: MyUnion) {}
