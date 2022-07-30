#![feature(unboxed_closures)]

fn foo<F: Fn(usize)>(_: F) {}
fn bar<F: Fn<usize>>(_: F) {}
fn main() {
    fn f(_: u64) {}
    foo(|_: isize| {}); //~ ERROR type mismatch
    bar(|_: isize| {}); //~ ERROR mismatched types
    foo(f); //~ ERROR type mismatch
    bar(f); //~ ERROR mismatched types
}
