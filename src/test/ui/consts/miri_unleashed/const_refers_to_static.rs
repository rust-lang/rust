// check-pass
// compile-flags: -Zunleash-the-miri-inside-of-you
#![warn(const_err)]

#![feature(const_raw_ptr_deref)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// Dynamically okay; does not touch any mutable static data:

const READ_IMMUT: &usize = {
    static FOO: usize = 0;
    &FOO
    //~^ WARN skipping const checks
};

const DEREF_IMMUT: usize = *READ_IMMUT;

const REF_INTERIOR_MUT: &usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { &*(&FOO as *const _ as *const usize) }
    //~^ WARN skipping const checks
};

extern { static EXTERN: usize; }
const REF_EXTERN: &usize = unsafe { &EXTERN };
//~^ WARN skipping const checks

// Not okay; uses of these consts would read or write mutable static data:

const DEREF_INTERIOR_MUT: usize = *REF_INTERIOR_MUT;
//~^ WARN any use of this value will cause an error

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

const READ_EXTERN: usize = unsafe { EXTERN }; //~ WARN any use of this value will cause an error
//~^ WARN skipping const checks

fn main() {}
