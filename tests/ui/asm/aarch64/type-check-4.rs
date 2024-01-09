// only-aarch64
// compile-flags: -C target-feature=+neon

#![feature(repr_simd, asm_const)]

use std::arch::aarch64::float64x2_t;
use std::arch::{asm, global_asm};

#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd256bit(f64, f64, f64, f64);

fn main() {
}

// Constants must be... constant

static S: i32 = 1;
const fn const_foo(x: i32) -> i32 {
    x
}
const fn const_bar<T>(x: T) -> T {
    x
}
global_asm!("{}", const S);
//~^ ERROR constants cannot refer to statics
global_asm!("{}", const const_foo(0));
global_asm!("{}", const const_foo(S));
//~^ ERROR constants cannot refer to statics
global_asm!("{}", const const_bar(0));
global_asm!("{}", const const_bar(S));
//~^ ERROR constants cannot refer to statics
