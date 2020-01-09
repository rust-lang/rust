// compile-flags: -Zunleash-the-miri-inside-of-you
#![warn(const_err)]

#![feature(const_raw_ptr_deref)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

const REF_INTERIOR_MUT: &usize = { //~ ERROR undefined behavior to use this value
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { &*(&FOO as *const _ as *const usize) }
    //~^ WARN skipping const checks
};

const MUTATE_INTERIOR_MUT: usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    FOO.fetch_add(1, Ordering::Relaxed) //~ WARN any use of this value will cause an error
    //~^ WARN skipping const checks
    //~| WARN skipping const checks
};

const READ_INTERIOR_MUT: usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { *(&FOO as *const _ as *const usize) } //~ WARN any use of this value will cause an err
    //~^ WARN skipping const checks
};

static mut MUTABLE: u32 = 0;
const READ_MUT: u32 = unsafe { MUTABLE }; //~ WARN any use of this value will cause an error
//~^ WARN skipping const checks
//~| WARN skipping const checks

// ok some day perhaps
const READ_IMMUT: &usize = { //~ ERROR it is undefined behavior to use this value
    static FOO: usize = 0;
    &FOO
    //~^ WARN skipping const checks
};
fn main() {}
