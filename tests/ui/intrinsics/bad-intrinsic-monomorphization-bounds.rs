//@ check-fail

#![feature(repr_simd, intrinsics, core_intrinsics)]
#![allow(warnings)]
#![crate_type = "rlib"]

// Bad monomorphizations could previously cause LLVM asserts even though the
// error was caught in the compiler.

use std::intrinsics;

#[derive(Copy, Clone)]
pub struct Foo(i64);

pub unsafe fn test_fadd_fast(a: Foo, b: Foo) -> Foo {
    intrinsics::fadd_fast(a, b)
    //~^ ERROR the trait bound `Foo: intrinsics::bounds::FloatPrimitive` is not satisfied
}
