//@ normalize-stderr-test: "\b10000(08|16|32)\b" -> "100$$PTR"
//@ normalize-stderr-test: "\b2500(060|120)\b" -> "250$$PTR"
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
    //~^ ERROR: this function may allocate
    let x = [0u8; 500_000];
    let x2 = [0u8; 500_000];
    let x3 = [0u8; 500_000];
    let x4 = [0u8; 500_000];
    let x5 = [0u8; 500_000];
    black_box((&x, &x2, &x3, &x4, &x5));
}

fn large_return_value() -> ArrayDefault<1_000_000> {
    //~^ ERROR: this function may allocate 1000000 bytes on the stack
    Default::default()
}

fn large_fn_arg(x: ArrayDefault<1_000_000>) {
    //~^ ERROR: this function may allocate
    black_box(&x);
}

fn has_large_closure() {
    let f = || black_box(&[0u8; 1_000_000]);
    //~^ ERROR: this function may allocate
    f();
}

fn main() {
    generic::<ArrayDefault<1_000_000>>();
}
