//@ check-pass

#![feature(precise_capturing_of_types)]
//~^ WARN the feature `precise_capturing_of_types` is incomplete

fn uncaptured<T>() -> impl Sized + use<> {}

fn main() {
    let mut x = uncaptured::<i32>();
    x = uncaptured::<u32>();
}
