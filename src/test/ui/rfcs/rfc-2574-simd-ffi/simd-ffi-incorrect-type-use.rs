// only-x86_64
// check-pass <- FIXME, see the below comment.

// The current stable rustc accepts this but it should fail.

#![allow(improper_ctypes, improper_ctypes_definitions)]

use std::arch::x86_64::__m128;

extern "C" {
    pub fn a(x: A);
    pub fn b(x: B);
}

#[repr(transparent)] pub struct A(__m128);
#[repr(C)] pub struct B(__m128);

pub extern "C" fn foo(x: __m128) -> __m128 { x }

fn main() {}
