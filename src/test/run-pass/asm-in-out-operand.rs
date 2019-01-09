#![feature(asm)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn next_power_of_2(n: u32) -> u32 {
    let mut tmp = n;
    asm!("dec $0" : "+rm"(tmp) :: "cc");
    let mut shift = 1_u32;
    while shift <= 16 {
        asm!(
            "shr %cl, $2
            or $2, $0
            shl $$1, $1"
            : "+&rm"(tmp), "+{ecx}"(shift) : "r"(tmp) : "cc"
        );
    }
    asm!("inc $0" : "+rm"(tmp) :: "cc");
    return tmp;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn main() {
    unsafe {
        assert_eq!(64, next_power_of_2(37));
        assert_eq!(2147483648, next_power_of_2(2147483647));
    }

    let mut y: isize = 5;
    let x: isize;
    unsafe {
        // Treat the output as initialization.
        asm!(
            "shl $2, $1
            add $3, $1
            mov $1, $0"
            : "=r"(x), "+r"(y) : "i"(3_usize), "ir"(7_usize) : "cc"
        );
    }
    assert_eq!(x, 47);
    assert_eq!(y, 47);

    let mut x = x + 1;
    assert_eq!(x, 48);

    unsafe {
        // Assignment to mutable.
        // Early clobber "&":
        // Forbids the use of a single register by both operands.
        asm!("shr $$2, $1; add $1, $0" : "+&r"(x) : "r"(x) : "cc");
    }
    assert_eq!(x, 60);
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn main() {}
