//@ normalize-stderr-32bit: "\(size: \d+, align: \d+\)" -> "(size: $$PTR, align: $$PTR)"
//@ normalize-stderr-64bit: "\(size: \d+, align: \d+\)" -> "(size: $$PTR, align: $$PTR)"
//@ normalize-stderr-test: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"

#![allow(static_mut_refs)]

const C1: &'static mut [usize] = &mut [];
//~^ ERROR: mutable references are not allowed

static mut S: i32 = 3;
const C2: &'static mut i32 = unsafe { &mut S };
//~^ ERROR: it is undefined behavior to use this value
//~| reference to mutable memory

fn main() {}
