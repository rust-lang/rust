//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(unboxed_closures)]

fn foo<F: Fn(usize)>(_: F) {}
fn bar<F: Fn<(usize,)>>(_: F) {}
fn main() {
    fn f(_: u64) {}
    foo(|_: isize| {}); //~ ERROR type mismatch
    bar(|_: isize| {}); //~ ERROR type mismatch
    foo(f); //~ ERROR type mismatch
    bar(f); //~ ERROR type mismatch
}
