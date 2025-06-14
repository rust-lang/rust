//@ normalize-stderr: "\(size: \d+, align: \d+\)" -> "(size: $$PTR, align: $$PTR)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE

#![allow(static_mut_refs)]

const C1: &'static mut [usize] = &mut [];
//~^ ERROR: mutable references are not allowed

fn main() {}
