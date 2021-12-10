// only-x86_64

#![allow(deprecated)] // llvm_asm!

fn main() {
    unsafe {
        println!("{:?}", llvm_asm!(""));
        //~^ ERROR prefer using the new asm! syntax instead
    }
}
