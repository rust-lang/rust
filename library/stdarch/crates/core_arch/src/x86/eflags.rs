//! `i386` intrinsics

use crate::arch::asm;

/// Reads EFLAGS.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=__readeflags)
#[cfg(target_arch = "x86")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.29.0",
    note = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __readeflags() -> u32 {
    let eflags: u32;
    asm!("pushfd", "pop {}", out(reg) eflags, options(nomem, att_syntax));
    eflags
}

/// Reads EFLAGS.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=__readeflags)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.29.0",
    note = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __readeflags() -> u64 {
    let eflags: u64;
    asm!("pushfq", "pop {}", out(reg) eflags, options(nomem, att_syntax));
    eflags
}

/// Write EFLAGS.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=__writeeflags)
#[cfg(target_arch = "x86")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.29.0",
    note = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __writeeflags(eflags: u32) {
    asm!("push {}", "popfd", in(reg) eflags, options(nomem, att_syntax));
}

/// Write EFLAGS.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=__writeeflags)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.29.0",
    note = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __writeeflags(eflags: u64) {
    asm!("push {}", "popfq", in(reg) eflags, options(nomem, att_syntax));
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;

    #[test]
    #[cfg_attr(miri, ignore)] // Uses inline assembly
    #[allow(deprecated)]
    fn test_readeflags() {
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
