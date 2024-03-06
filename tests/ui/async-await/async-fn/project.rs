//@ edition:2018
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(async_closure, unboxed_closures, async_fn_traits)]

fn project<F: async Fn<()>>(_: F) -> Option<F::Output> { None }

fn main() {
    let x: Option<i32> = project(|| async { 1i32 });
}
