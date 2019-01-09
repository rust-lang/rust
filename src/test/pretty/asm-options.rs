#![feature(asm)]

// pp-exact

pub fn main() {
    unsafe {
        asm!("" : : : : "volatile");
        asm!("" : : : : "alignstack");
        asm!("" : : : : "intel");
    }
}
