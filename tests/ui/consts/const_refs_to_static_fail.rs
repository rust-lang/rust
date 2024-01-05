#![feature(const_refs_to_static, const_mut_refs, sync_unsafe_cell)]
use std::cell::SyncUnsafeCell;

static S: SyncUnsafeCell<i32> = SyncUnsafeCell::new(0);
static mut S_MUT: i32 = 0;

const C1: &SyncUnsafeCell<i32> = &S;
const C1_READ: () = unsafe {
    assert!(*C1.get() == 0); //~ERROR evaluation of constant value failed
    //~^ constant accesses mutable global memory
};
const C2: *const i32 = unsafe { std::ptr::addr_of!(S_MUT) };
const C2_READ: () = unsafe {
    assert!(*C2 == 0); //~ERROR evaluation of constant value failed
    //~^ constant accesses mutable global memory
};

fn main() {
}
