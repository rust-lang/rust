#![feature(unboxed_closures)]

fn foo<F: Fn<i32>>(f: F) -> F::Output { f(3) } //~ ERROR E0059

fn main() {
}
