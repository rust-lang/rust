//issue: rust-lang/rust#122548

// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"

use std::cell::UnsafeCell;

struct Meh {
    x: &'static UnsafeCell<i32>,
}

const MUH: Meh = Meh { x: unsafe { &mut *(&READONLY as *const _ as *mut _) } };
//~^ ERROR: invalid value at .x.<deref>: encountered `UnsafeCell` in read-only memory

static READONLY: i32 = 0;

pub fn main() {}
