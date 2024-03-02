// This test verifies the accuracy of emitted file and line debuginfo metadata for closures and
// coroutines.
//
//@ compile-flags: -C debuginfo=2 -Z debug-info-type-line-numbers=true
#![crate_type = "lib"]
#![feature(coroutines, stmt_expr_attributes)]

// ignore-tidy-linelength

// NONMSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}/issue-98678-closure-coroutine.rs{{".*}})
// MSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\issue-98678-closure-coroutine.rs{{".*}})

pub fn foo() {
    // NONMSVC: !DICompositeType({{.*"}}{closure_env#0}{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // MSVC-DAG: !DICompositeType({{.*"}}closure_env$0{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let closure = |x| x;
    closure(0);

    // NONMSVC: !DICompositeType({{.*"[{]}}coroutine_env#1{{[}]".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // MSVC-DAG: !DICompositeType({{.*".*foo::}}coroutine_env$1>{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let coroutine = #[coroutine]
    || yield 1;
}
