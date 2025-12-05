#![feature(unboxed_closures)]

fn a<F: Fn<usize>>(f: F) {}
//~^ ERROR type parameter to bare `Fn` trait must be a tuple

fn main() {
    a(|_: usize| {}); //~ ERROR: mismatched types
}
