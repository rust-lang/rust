// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code, warnings)]

static mut x: isize = 3;
static mut y: isize = unsafe { x };

fn main() {}
