// This test verifies the accuracy of emitted file and line debuginfo metadata for closures and
// coroutines.
//
//@ compile-flags: --crate-type=lib -Copt-level=0 -Cdebuginfo=2 -Zdebug-info-type-line-numbers=true
#![feature(coroutines, stmt_expr_attributes)]

// ignore-tidy-linelength

// NONMSVC-DAG: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*[/\\]}}issue-98678-closure-coroutine.rs{{".*}})
// MSVC-DAG: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\issue-98678-closure-coroutine.rs{{".*}})

pub fn foo() {
    // NONMSVC-DAG: !DICompositeType({{.*"}}{closure_env#0}{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // MSVC-DAG: !DICompositeType({{.*"}}closure_env$0{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let closure = |x| x;
    closure(0);

    // NONMSVC-DAG: !DICompositeType({{.*"[{]}}coroutine_env#1{{[}]".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 3]],
    // MSVC-DAG: !DICompositeType({{.*".*foo::}}coroutine_env$1>{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    let _coroutine = #[coroutine]
    || yield 1;
}
