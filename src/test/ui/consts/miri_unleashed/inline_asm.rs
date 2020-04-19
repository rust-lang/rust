// compile-flags: -Zunleash-the-miri-inside-of-you
// only-x86_64
#![feature(llvm_asm)]
#![allow(const_err)]

fn main() {}

// Make sure we catch executing inline assembly.
static TEST_BAD: () = {
    unsafe { llvm_asm!("xor %eax, %eax" ::: "eax"); }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE in this expansion of llvm_asm!
    //~| NOTE inline assembly is not supported
};
