//! Access types available on all architectures

/// Full system is the required shareability domain, reads and writes are the
/// required access types
#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
pub struct SY;

dmb_dsb!(SY);

#[unstable(feature = "stdarch_arm_barrier", issue = "117219")]
impl super::super::sealed::Isb for SY {
    #[inline(always)]
    unsafe fn __isb(&self) {
        super::isb(super::arg::SY)
    }
}
