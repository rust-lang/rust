// This test verifies the accuracy of emitted file and line debuginfo metadata for closures and
// generators.
//
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]
#![feature(generators, stmt_expr_attributes)]

// CHECK: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}/codegen/issue-98678-closure-generator.rs{{".*}})

pub fn foo() {
    // CHECK: !DICompositeType({{.*"[{]}}closure_env#0{{[}]".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let closure = |x| x;
    closure(0);

    // CHECK: !DICompositeType({{.*"[{]}}generator_env#1{{[}]".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    let generator = #[coroutine]
    || yield 1;
}
