// ignore-tidy-linelength
//! This test verifies the accuracy of emitted file and line debuginfo metadata for async blocks and
//! async functions.

//@ revisions: MSVC NONMSVC
//@[MSVC] only-msvc
//@[NONMSVC] ignore-msvc
//@ edition:2021
//@ compile-flags: --crate-type=lib -Copt-level=0 -Cdebuginfo=2 -Zdebug-info-type-line-numbers=true

// NONMSVC-DAG: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*[/\\]}}issue-98678-async.rs{{".*}})
// MSVC: ![[#FILE:]] = !DIFile({{.*}}filename:{{.*}}\\issue-98678-async.rs{{".*}})

// NONMSVC-DAG: !DISubprogram(name: "foo",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
// MSVC-DAG: !DISubprogram(name: "foo",{{.*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
pub async fn foo() -> u8 {
    5
}

pub fn bar() -> impl std::future::Future<Output = u8> {
    // NONMSVC: !DICompositeType({{.*"}}{async_block_env#0}{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 2]],
    // MSVC-DAG: !DICompositeType({{.*"}}enum2$<issue_98678_async::bar::async_block_env$0>{{".*}}file: ![[#FILE]]{{.*}}line: [[# @LINE + 1]],
    async {
        let x: u8 = foo().await;
        x + 5
    }
}
