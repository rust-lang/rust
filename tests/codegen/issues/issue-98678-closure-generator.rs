// This test verifies the accuracy of emitted file and line debuginfo metadata for closures and
// generators.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]
#![feature(generators, stmt_expr_attributes)]

// ignore-tidy-linelength

// NONMSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}/codegen/issue-98678-closure-generator.rs{{".*}})
// MSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\codegen\\issue-98678-closure-generator.rs{{".*}})

pub fn foo() {
    // NONMSVC: !DICompositeType({{.*"}}{closure_env#0}{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // MSVC-DAG: !DICompositeType({{.*"}}closure_env$0{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let closure = |x| x;
    closure(0);

    // NONMSVC: !DICompositeType({{.*"[{]}}generator_env#1{{[}]".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // MSVC-DAG: !DICompositeType({{.*".*foo::}}generator_env$1>{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let generator = #[coroutine]
    || yield 1;
}
