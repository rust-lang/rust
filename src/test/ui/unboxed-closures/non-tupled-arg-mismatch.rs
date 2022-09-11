#![feature(unboxed_closures)]

fn a<F: Fn<usize>>(f: F) {}

fn main() {
    a(|_: usize| {});
    //~^ ERROR mismatched types
}
