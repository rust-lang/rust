#![feature(const_fn)]

use std::mem;

#[repr(transparent)]
struct Foo(u32);

const TRANSMUTED_U32: u32 = unsafe { mem::transmute(Foo(3)) };

const fn transmute_fn() -> u32 { unsafe { mem::transmute(Foo(3)) } }
//~^ ERROR The use of std::mem::transmute() is gated in const fns

fn main() {}
