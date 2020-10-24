// run-pass
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

fn with_bound<const N: usize>() where [u8; N / 2]: Sized {
    let _: [u8; N / 2] = [0; N / 2];
}

fn main() {
    with_bound::<4>();
}
