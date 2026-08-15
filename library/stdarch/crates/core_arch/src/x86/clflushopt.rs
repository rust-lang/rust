//! `CLFLUSHOPT` cache-line flush.

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.clflushopt"]
    fn clflushopt(p: *const u8);
}

/// Invalidates from every level of the cache hierarchy the cache line that
/// contains `p`.
///
/// Unlike [`_mm_clflush`], `CLFLUSHOPT` is only ordered with respect to older
/// writes to the flushed cache line and with respect to fence/locked
/// operations; it is *not* serialized against other `CLFLUSHOPT`/`CLFLUSH`
/// instructions or unrelated stores. This makes flushing a range of lines
/// substantially faster, but a fence (e.g. [`_mm_sfence`] or [`_mm_mfence`]) is
/// required afterward to order the flushes against subsequent operations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_clflushopt)
///
/// # Safety
///
/// Unlike the prefetch intrinsics, `CLFLUSHOPT` is subject to all the
/// permission checking and faults associated with a byte load, so `p` must
/// point to a byte that is valid for reads.
///
/// [`_mm_clflush`]: crate::arch::x86::_mm_clflush
/// [`_mm_sfence`]: crate::arch::x86::_mm_sfence
/// [`_mm_mfence`]: crate::arch::x86::_mm_mfence
#[inline]
#[target_feature(enable = "clflushopt")]
#[cfg_attr(test, assert_instr(clflushopt))]
#[unstable(feature = "simd_x86_clflushopt", issue = "157096")]
pub unsafe fn _mm_clflushopt(p: *const u8) {
    clflushopt(p);
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "clflushopt")]
    unsafe fn test_mm_clflushopt() {
        let x = 0_u8;
        _mm_clflushopt(core::ptr::addr_of!(x));
    }
}
