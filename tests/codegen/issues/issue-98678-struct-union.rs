// ignore-tidy-linelength
//! This test verifies the accuracy of emitted file and line debuginfo metadata for structs and
//! unions.

//@ revisions: MSVC NONMSVC
//@[MSVC] only-msvc
//@[NONMSVC] ignore-msvc
//@ compile-flags: --crate-type=lib -Copt-level=0 -Cdebuginfo=2 -Zdebug-info-type-line-numbers=true

// NONMSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*[/\\]}}issue-98678-struct-union.rs{{".*}})
// MSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\issue-98678-struct-union.rs{{".*}})

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
