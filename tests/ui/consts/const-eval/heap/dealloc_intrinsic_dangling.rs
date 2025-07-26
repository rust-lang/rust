#![feature(core_intrinsics)]
#![feature(const_heap)]

// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ normalize-stderr: "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"

use std::intrinsics;

const _X: &'static u8 = unsafe {
    //~^ ERROR: dangling reference (use-after-free)
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 4);
    &*ptr
};

const _Y: u8 = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    let reference = &*ptr;
    intrinsics::const_deallocate(ptr, 4, 4);
    *reference
    //~^ ERROR: has been freed, so this pointer is dangling
};

fn main() {}
