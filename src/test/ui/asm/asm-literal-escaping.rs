// build-pass
// only-x86_64

#![feature(asm)]

fn main() {
    unsafe {
        // "nop" :: "r"(x) : "eax" : "volatile"
        let x = 10;
        asm!("\x6Eop" :: "\x72"(x) : "\x65ax" : "\x76olatile");
    }
}
