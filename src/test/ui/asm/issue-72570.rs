#![feature(asm)]

fn main() {
    unsafe {
        asm!("", in("invalid") "".len());
    }
}
