//@ add-core-stubs
//@ revisions:x86-linux x86-darwin

//@[x86-linux] compile-flags: --target i686-unknown-linux-gnu
//@[x86-linux] needs-llvm-components: x86
//@[x86-darwin] compile-flags: --target i686-apple-darwin
//@[x86-darwin] needs-llvm-components: x86

// Tests that aggregates containing vector types get their alignment increased to 16 on Darwin.

#![feature(no_core, repr_simd, simd_ffi)]
#![crate_type = "lib"]
#![no_std]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct i32x4([i32; 4]);

#[repr(C)]
pub struct Foo {
    a: i32x4,
    b: i8,
}

// This tests that we recursively check for vector types, not just at the top level.
#[repr(C)]
pub struct DoubleFoo {
    one: Foo,
    two: Foo,
}

extern "C" {
    // x86-linux: declare void @f({{.*}}byval([32 x i8]) align 4{{.*}})
    // x86-darwin: declare void @f({{.*}}byval([32 x i8]) align 16{{.*}})
    fn f(foo: Foo);

    // x86-linux: declare void @g({{.*}}byval([64 x i8]) align 4{{.*}})
    // x86-darwin: declare void @g({{.*}}byval([64 x i8]) align 16{{.*}})
    fn g(foo: DoubleFoo);
}

pub fn main() {
    unsafe { f(Foo { a: i32x4([1, 2, 3, 4]), b: 0 }) }

    unsafe {
        g(DoubleFoo {
            one: Foo { a: i32x4([1, 2, 3, 4]), b: 0 },
            two: Foo { a: i32x4([1, 2, 3, 4]), b: 0 },
        })
    }
}
