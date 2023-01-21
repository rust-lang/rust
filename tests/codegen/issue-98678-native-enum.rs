// This test verifies the accuracy of emitted file and line debuginfo metadata for native
// enumerations.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]

// CHECK: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}/codegen/issue-98678-native-enum.rs{{".*}})

// CHECK: !DICompositeType({{.*"}}MyNativeEnum{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
// CHECK: !DICompositeType({{.*}}DW_TAG_variant_part{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub enum MyNativeEnum {
    // CHECK: !DIDerivedType({{.*"}}One{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    One,
}

pub fn foo(_: MyNativeEnum) {}
