#![feature(core_intrinsics)]
#![feature(const_heap)]

// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr-test: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ normalize-stderr-test: "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"

use std::intrinsics;

const _X: &'static u8 = unsafe {
    //~^ error: it is undefined behavior to use this value
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 4);
    &*ptr
};

const _Y: u8 = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    let reference = &*ptr;
    intrinsics::const_deallocate(ptr, 4, 4);
    *reference
    //~^ error: evaluation of constant value failed
};

fn main() {}
