#![feature(repr_simd, platform_intrinsics, core_intrinsics)]
#![allow(warnings)]
#![crate_type = "rlib"]

// Bad monomorphizations could previously cause LLVM asserts even though the
// error was caught in the compiler.

extern "platform-intrinsic" {
    fn simd_add<T>(x: T, y: T) -> T;
}

use std::intrinsics;

#[derive(Copy, Clone)]
pub struct Foo(i64);

pub fn test_cttz(v: Foo) -> Foo {
    intrinsics::cttz(v)
    //~^ ERROR `cttz` intrinsic: expected basic integer type, found `Foo`
}

pub unsafe fn test_fadd_fast(a: Foo, b: Foo) -> Foo {
    intrinsics::fadd_fast(a, b)
    //~^ ERROR `fadd_fast` intrinsic: expected basic float type, found `Foo`
}

pub unsafe fn test_simd_add(a: Foo, b: Foo) -> Foo {
    simd_add(a, b)
    //~^ ERROR `simd_add` intrinsic: expected SIMD input type, found non-SIMD `Foo`
}
