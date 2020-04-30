// build-fail
// compile-flags: -Zunleash-the-miri-inside-of-you
#![allow(const_err)]

#![feature(const_raw_ptr_deref)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// These fail during CTFE (as they read a static), so they only cause an error
// when *using* the const.

const MUTATE_INTERIOR_MUT: usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    FOO.fetch_add(1, Ordering::Relaxed)
    //~^ WARN skipping const checks
    //~| WARN skipping const checks
};

const READ_INTERIOR_MUT: usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { *(&FOO as *const _ as *const usize) }
    //~^ WARN skipping const checks
};

static mut MUTABLE: u32 = 0;
const READ_MUT: u32 = unsafe { MUTABLE };
//~^ WARN skipping const checks
//~| WARN skipping const checks

fn main() {
    MUTATE_INTERIOR_MUT;
    //~^ ERROR: erroneous constant used
    READ_INTERIOR_MUT;
    //~^ ERROR: erroneous constant used
    READ_MUT;
    //~^ ERROR: erroneous constant used
}
