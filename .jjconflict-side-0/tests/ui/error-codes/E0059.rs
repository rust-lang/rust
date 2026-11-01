#![feature(unboxed_closures)]

fn foo<F: Fn<i32>>(f: F) -> F::Output { f(3) } //~ ERROR E0059
//~^ ERROR `i32` is not a tuple
//~| ERROR cannot use call notation

fn main() {
}
