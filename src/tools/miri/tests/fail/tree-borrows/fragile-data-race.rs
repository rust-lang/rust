//! Race-condition-like interaction between a read and a reborrow.
//! Even though no write or fake write occurs, reads have an effect on protected
//! Reserved. This is a protected-retag/read data race, but is not *detected* as
//! a data race violation because reborrows are not writes.
//!
//! This test is sensitive to the exact schedule so we disable preemption.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-preemption-rate=0
use std::ptr::addr_of_mut;
use std::thread;

#[derive(Copy, Clone)]
struct SendPtr(*mut u8);

unsafe impl Send for SendPtr {}

// First thread is just a reborrow, but for an instant `x` is
// protected and thus vulnerable to foreign reads.
fn thread_1(x: &mut u8) -> SendPtr {
    thread::yield_now(); // make the other thread go first
    SendPtr(x as *mut u8)
}

// Second thread simply performs a read.
fn thread_2(x: &u8) {
    let _val = *x;
}

fn main() {
    let mut x = 0u8;
    let x_1 = unsafe { &mut *addr_of_mut!(x) };
    let xg = unsafe { &*addr_of_mut!(x) };

    // The two threads are executed in parallel on aliasing pointers.
    // UB occurs if the read of thread_2 occurs while the protector of thread_1
    // is in place.
    let hf = thread::spawn(move || thread_1(x_1));
    let hg = thread::spawn(move || thread_2(xg));
    let SendPtr(p) = hf.join().unwrap();
    let () = hg.join().unwrap();

    unsafe { *p = 1 }; //~ ERROR: /write access through .* is forbidden/
}
