//@ only-aarch64
//@ compile-flags: -C target-feature=+neon

#![feature(repr_simd)]

use std::arch::aarch64::float64x2_t;
use std::arch::{asm, global_asm};

#[repr(simd)]
#[derive(Copy, Clone)]
struct Simd256bit([f64; 4]);

fn main() {}

// Constants must be... constant

static S: i32 = 1;
const fn const_foo(x: i32) -> i32 {
    x
}
const fn const_bar<T>(x: T) -> T {
    x
}
global_asm!("{}", const S);
//~^ ERROR referencing statics
global_asm!("{}", const const_foo(0));
global_asm!("{}", const const_foo(S));
//~^ ERROR referencing statics
global_asm!("{}", const const_bar(0));
global_asm!("{}", const const_bar(S));
//~^ ERROR referencing statics
