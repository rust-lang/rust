// This test verifies the accuracy of emitted file and line debuginfo metadata for C++-like
// enumerations.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]

// The use of CHECK-DAG here is because the C++-like enum is emitted before the `DIFile` node

// CHECK-DAG: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}/codegen/issue-98678-cpp-like-enum.rs{{".*}})

// CHECK-DAG: !DICompositeType({{.*"}}MyCppLikeEnum{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
#[repr(C)]
pub enum MyCppLikeEnum {
    One,
}

pub fn foo(_: MyCppLikeEnum) {}
