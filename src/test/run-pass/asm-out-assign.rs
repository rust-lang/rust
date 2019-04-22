#![feature(asm)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn main() {
    let x: isize;
    unsafe {
        // Treat the output as initialization.
        asm!("mov $1, $0" : "=r"(x) : "r"(5_usize));
    }
    assert_eq!(x, 5);

    let mut x = x + 1;
    assert_eq!(x, 6);

    unsafe {
        // Assignment to mutable.
        asm!("mov $1, $0" : "=r"(x) : "r"(x + 7));
    }
    assert_eq!(x, 13);
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn main() {}
