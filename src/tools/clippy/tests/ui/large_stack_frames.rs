#![allow(unused, incomplete_features)]
#![warn(clippy::large_stack_frames)]
#![feature(unsized_locals)]

use std::hint::black_box;

fn generic<T: Default>() {
    let x = T::default();
    black_box(&x);
}

fn unsized_local() {
    let x: dyn std::fmt::Display = *(Box::new(1) as Box<dyn std::fmt::Display>);
    black_box(&x);
}

struct ArrayDefault<const N: usize>([u8; N]);

impl<const N: usize> Default for ArrayDefault<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

fn many_small_arrays() {
    let x = [0u8; 500_000];
    let x2 = [0u8; 500_000];
    let x3 = [0u8; 500_000];
    let x4 = [0u8; 500_000];
    let x5 = [0u8; 500_000];
    black_box((&x, &x2, &x3, &x4, &x5));
}

fn large_return_value() -> ArrayDefault<1_000_000> {
    Default::default()
}

fn large_fn_arg(x: ArrayDefault<1_000_000>) {
    black_box(&x);
}

fn main() {
    generic::<ArrayDefault<1_000_000>>();
}
