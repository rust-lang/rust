#![feature(llvm_asm)]

// pp-exact

pub fn main() {
    unsafe {
        llvm_asm!("" : : : : "volatile");
        llvm_asm!("" : : : : "alignstack");
        llvm_asm!("" : : : : "intel");
    }
}
