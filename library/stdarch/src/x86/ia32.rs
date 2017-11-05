//! `i386/ia32` intrinsics

/// Reads EFLAGS.
#[cfg(target_arch = "x86")]
#[inline(always)]
pub unsafe fn __readeflags() -> u32 {
    let eflags: u32;
    asm!("pushfd; popl $0" : "=r"(eflags) : : : "volatile");
    eflags
}

/// Reads EFLAGS.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn __readeflags() -> u64 {
    let eflags: u64;
    asm!("pushfq; popq $0" : "=r"(eflags) : : : "volatile");
    eflags
}

/// Write EFLAGS.
#[cfg(target_arch = "x86")]
#[inline(always)]
pub unsafe fn __writeeflags(eflags: u32) {
    asm!("pushl $0; popfd" : : "r"(eflags) : "cc", "flags" : "volatile");
}

/// Write EFLAGS.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn __writeeflags(eflags: u64) {
    asm!("pushq $0; popfq" : : "r"(eflags) : "cc", "flags" : "volatile");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eflags() {
        unsafe {
            // reads eflags, writes them back, reads them again,
            // and compare for equality:
            let v = __readeflags();
            __writeeflags(v);
            let u = __readeflags();
            assert_eq!(v, u);
        }
    }
}
