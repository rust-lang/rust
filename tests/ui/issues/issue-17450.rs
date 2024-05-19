//@ build-pass
#![allow(dead_code, warnings)]

static mut x: isize = 3;
static mut y: isize = unsafe { x };

fn main() {}
