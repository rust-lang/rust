#![feature(box_syntax)]

static mut a: Box<isize> = box 3;
//~^ ERROR allocations are not allowed in statics

fn main() {}
