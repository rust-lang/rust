// ignore-tidy-linelength
//! This test verifies the accuracy of emitted file and line debuginfo metadata enums.

//@ revisions: MSVC NONMSVC
//@[MSVC] only-msvc
//@[NONMSVC] ignore-msvc
//@ compile-flags: --crate-type=lib -Copt-level=0 -Cdebuginfo=2 -Zdebug-info-type-line-numbers=true

// NONMSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*[/\\]}}issue-98678-enum.rs{{".*}})
// MSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\issue-98678-enum.rs{{".*}})

// NONMSVC: !DICompositeType({{.*"}}SingleCase{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
// MSVC: !DICompositeType({{.*"}}enum2$<issue_98678_enum::SingleCase>{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub enum SingleCase {
    // NONMSVC: !DIDerivedType(tag: DW_TAG_member, name: "One",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "One",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    One,
}

// NONMSVC: !DICompositeType({{.*"}}MultipleDataCases{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
// MSVC: !DICompositeType({{.*"}}enum2$<issue_98678_enum::MultipleDataCases>{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub enum MultipleDataCases {
    // NONMSVC: !DIDerivedType(tag: DW_TAG_member, name: "Case1",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Case1",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    Case1(u32),
    // NONMSVC: !DIDerivedType(tag: DW_TAG_member, name: "Case2",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Case2",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    Case2(i64),
}

// NONMSVC: !DICompositeType({{.*"}}NicheLayout{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
// MSVC: !DICompositeType({{.*"}}enum2$<issue_98678_enum::NicheLayout>{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub enum NicheLayout {
    // NONMSVC: !DIDerivedType(tag: DW_TAG_member, name: "Something",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Something",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    Something(&'static u32),
    // NONMSVC: !DIDerivedType(tag: DW_TAG_member, name: "Nothing",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Nothing",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    Nothing,
}

pub fn foo(_: SingleCase, _: MultipleDataCases, _: NicheLayout) {}
