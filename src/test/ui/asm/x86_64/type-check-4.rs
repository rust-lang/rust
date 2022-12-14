// only-x86_64
// compile-flags: -C target-feature=+avx512f

#![feature(asm_const)]

use std::arch::{asm, global_asm};

use std::arch::x86_64::{_mm256_setzero_ps, _mm_setzero_ps};

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
//~^ ERROR constants cannot refer to statics
global_asm!("{}", const const_foo(0));
global_asm!("{}", const const_foo(S));
//~^ ERROR constants cannot refer to statics
global_asm!("{}", const const_bar(0));
global_asm!("{}", const const_bar(S));
//~^ ERROR constants cannot refer to statics
