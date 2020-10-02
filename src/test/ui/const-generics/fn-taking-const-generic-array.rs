// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

use std::fmt::Display;

fn print_slice<T: Display, const N: usize>(slice: &[T; N]) {
    for x in slice.iter() {
        println!("{}", x);
    }
}

fn main() {
    print_slice(&[1, 2, 3]);
}
