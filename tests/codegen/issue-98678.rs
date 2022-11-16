// This test verifies the accuracy of emitted file and line debuginfo metadata.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]

// CHECK: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}src/test/codegen/issue-98678.rs{{".*}})

// CHECK: !DICompositeType({{.*"}}MyType{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub struct MyType;

pub fn foo(_: MyType) {}
