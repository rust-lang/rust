#![feature(const_transmute)]

use std::mem;

static FOO: bool = unsafe { mem::transmute(3u8) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
