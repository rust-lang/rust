// This test verifies the accuracy of emitted file and line debuginfo metadata for native
// enumerations.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]

// ignore-tidy-linelength

// NONMSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}/codegen/issue-98678-native-enum.rs{{".*}})
// MSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\codegen\\issue-98678-native-enum.rs{{".*}})

// NONMSVC: !DICompositeType({{.*"}}MyNativeEnum{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 4]],
// MSVC: !DICompositeType({{.*::}}MyNativeEnum>{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 3]],
// NONMSVC: !DICompositeType({{.*}}DW_TAG_variant_part{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
// COM: MSVC: currently no DW_TAG_variant_part from MSVC
pub enum MyNativeEnum {
    // NONMSVC: !DIDerivedType({{.*"}}One{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // MSVC: !DICompositeType({{.*"}}One{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    One,
}

pub fn foo(_: MyNativeEnum) {}
