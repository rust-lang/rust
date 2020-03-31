// build-pass
// only-x86_64

#![feature(llvm_asm)]

fn main() {
    unsafe {
        // "nop" :: "r"(x) : "eax" : "volatile"
        let x = 10;
        llvm_asm!("\x6Eop" :: "\x72"(x) : "\x65ax" : "\x76olatile");
    }
}
