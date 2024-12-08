//@ compile-flags: -Zunleash-the-miri-inside-of-you
//@ normalize-stderr-test: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test: "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"

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

// Evaluating this does not read anything mutable, but validation does, so this should error.
const REF_INTERIOR_MUT: &usize = { //~ ERROR undefined behavior
    //~| encountered reference to mutable memory
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { &*(&FOO as *const _ as *const usize) }
};

// Not actually reading from anything mutable, so these are fine.
static MY_STATIC: u8 = 4;
const REF_IMMUT: &u8 = &MY_STATIC;
const READ_IMMUT: u8 = *REF_IMMUT;

fn main() {}
