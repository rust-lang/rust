// build-fail
// compile-flags: -Zunleash-the-miri-inside-of-you -Zdeduplicate-diagnostics
#![allow(const_err)]
#![feature(const_raw_ptr_deref)] // FIXME: cannot remove because then rustc thinks there is no error
#![crate_type = "lib"]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// These fail during CTFE (as they read a static), so they only cause an error
// when *using* the const.

const MUTATE_INTERIOR_MUT: usize = {
//~^ WARN skipping const checks
    static FOO: AtomicUsize = AtomicUsize::new(0);
    FOO.fetch_add(1, Ordering::Relaxed)
};

const READ_INTERIOR_MUT: usize = {
//~^ WARN skipping const checks
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { *(&FOO as *const _ as *const usize) }
};

static mut MUTABLE: u32 = 0;
const READ_MUT: u32 = unsafe { MUTABLE };
//~^ WARN skipping const checks

pub fn main() {
    MUTATE_INTERIOR_MUT;
    //~^ ERROR: erroneous constant used
    READ_INTERIOR_MUT;
    //~^ ERROR: erroneous constant used
    READ_MUT;
    //~^ ERROR: erroneous constant used
}
