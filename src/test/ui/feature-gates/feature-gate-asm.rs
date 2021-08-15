// only-x86_64

#![allow(deprecated)] // llvm_asm!

fn main() {
    unsafe {
        asm!("");
        //~^ ERROR inline assembly is not stable enough
        llvm_asm!("");
        //~^ ERROR prefer using the new asm! syntax instead
    }
}
