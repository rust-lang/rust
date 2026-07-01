//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ normalize-stderr: "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

struct RefDst {
    b: unsafe<'a> &'a u32,
}

const C1: &RefDst = unsafe { std::mem::transmute(&1usize) };
//~^ ERROR: encountered a dangling reference

fn main() {}
