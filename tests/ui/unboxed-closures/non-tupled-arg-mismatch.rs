//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(unboxed_closures)]

fn a<F: Fn<usize>>(f: F) {}
//~^ ERROR type parameter to bare `Fn` trait must be a tuple

fn main() {
    a(|_: usize| {}); //~ ERROR: mismatched types
}
