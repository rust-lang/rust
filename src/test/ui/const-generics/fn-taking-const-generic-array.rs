// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

use std::fmt::Display;

fn print_slice<T: Display, const N: usize>(slice: &[T; N]) {
    for x in slice.iter() {
        println!("{}", x);
    }
}

fn main() {
    print_slice(&[1, 2, 3]);
}
