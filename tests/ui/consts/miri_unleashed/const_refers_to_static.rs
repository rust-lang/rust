// compile-flags: -Zunleash-the-miri-inside-of-you
// stderr-per-bitwidth

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

const MUTATE_INTERIOR_MUT: usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    FOO.fetch_add(1, Ordering::Relaxed) //~ERROR evaluation of constant value failed
};

const READ_INTERIOR_MUT: usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { *(&FOO as *const _ as *const usize) } //~ERROR evaluation of constant value failed
};

static mut MUTABLE: u32 = 0;
const READ_MUT: u32 = unsafe { MUTABLE }; //~ERROR evaluation of constant value failed

// Not actually reading from anything mutable, so these are fine.
const REF_INTERIOR_MUT: &usize = {
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { &*(&FOO as *const _ as *const usize) }
};

static MY_STATIC: u8 = 4;
const REF_IMMUT: &u8 = &MY_STATIC;
const READ_IMMUT: u8 = *REF_IMMUT;

fn main() {}
