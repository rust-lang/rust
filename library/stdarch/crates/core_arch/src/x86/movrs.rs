//! Read-shared move intrinsics

#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.prefetchrs"]
    fn prefetchrs(p: *const u8);
}

/// Prefetches the cache line that contains address `p`, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
///
/// Note: this intrinsic is safe to use even though it takes a raw pointer argument. In general, this
/// cannot change the behavior of the program, including not trapping on invalid pointers.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(prefetchrst2))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub fn _m_prefetchrs(p: *const u8) {
    unsafe { prefetchrs(p) }
}
