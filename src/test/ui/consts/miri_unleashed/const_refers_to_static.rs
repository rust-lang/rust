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

const REF_INTERIOR_MUT: &usize = { //~ ERROR undefined behavior to use this value
//~| encountered a reference pointing to a static variable
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { &*(&FOO as *const _ as *const usize) }
};

// ok some day perhaps
const READ_IMMUT: &usize = { //~ ERROR it is undefined behavior to use this value
//~| encountered a reference pointing to a static variable
    static FOO: usize = 0;
    &FOO
};

fn main() {}
