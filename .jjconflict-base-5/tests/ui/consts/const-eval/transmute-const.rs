#![allow(unnecessary_transmutes)]
use std::mem;

static FOO: bool = unsafe { mem::transmute(3u8) };
//~^ ERROR 0x03, but expected a bool

fn main() {}
