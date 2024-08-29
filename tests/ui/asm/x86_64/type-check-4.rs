//@ only-x86_64
//@ compile-flags: -C target-feature=+avx512f
//@ check-pass

use std::arch::x86_64::{_mm256_setzero_ps, _mm_setzero_ps};
use std::arch::{asm, global_asm};

fn main() {}

static S: i32 = 1;
const fn const_foo(x: i32) -> i32 {
    x
}
const fn const_bar<T>(x: T) -> T {
    x
}
global_asm!("{}", const S);
global_asm!("{}", const const_foo(0));
global_asm!("{}", const const_foo(S));
global_asm!("{}", const const_bar(0));
global_asm!("{}", const const_bar(S));
