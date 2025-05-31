//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE

#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;

static S: SyncUnsafeCell<i32> = SyncUnsafeCell::new(0);
static mut S_MUT: i32 = 0;

const C1: &SyncUnsafeCell<i32> = &S; //~ERROR encountered reference to mutable memory
const C1_READ: () = unsafe {
    assert!(*C1.get() == 0);
};
const C2: *const i32 = unsafe { std::ptr::addr_of!(S_MUT) };
const C2_READ: () = unsafe {
    assert!(*C2 == 0); //~ERROR constant accesses mutable global memory
};

fn main() {}
