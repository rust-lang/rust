//@ normalize-stderr-test: "\b10000(08|16|32)\b" -> "100$$PTR"
//@ normalize-stderr-test: "\b2500(060|120)\b" -> "250$$PTR"
#![allow(unused)]
#![warn(clippy::large_stack_frames)]

use std::hint::black_box;

fn generic<T: Default>() {
    let x = T::default();
    black_box(&x);
}

struct ArrayDefault<const N: usize>([u8; N]);

impl<const N: usize> Default for ArrayDefault<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

fn many_small_arrays() {
    //~^ large_stack_frames

    let x = [0u8; 500_000];
    let x2 = [0u8; 500_000];
    let x3 = [0u8; 500_000];
    let x4 = [0u8; 500_000];
    let x5 = [0u8; 500_000];
    black_box((&x, &x2, &x3, &x4, &x5));
}

fn large_return_value() -> ArrayDefault<1_000_000> {
    //~^ large_stack_frames

    Default::default()
}

fn large_fn_arg(x: ArrayDefault<1_000_000>) {
    //~^ large_stack_frames

    black_box(&x);
}

fn has_large_closure() {
    let f = || black_box(&[0u8; 1_000_000]);
    //~^ large_stack_frames

    f();
}

fn main() {
    generic::<ArrayDefault<1_000_000>>();
}
