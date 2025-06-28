//! RDTSC instructions.

#[cfg(test)]
use stdarch_test::assert_instr;

/// Reads the current value of the processor’s time-stamp counter.
///
/// The processor monotonically increments the time-stamp counter MSR
/// every clock cycle and resets it to 0 whenever the processor is
/// reset.
///
/// The RDTSC instruction is not a serializing instruction. It does
/// not necessarily wait until all previous instructions have been
/// executed before reading the counter. Similarly, subsequent
/// instructions may begin execution before the read operation is
/// performed.
///
/// On processors that support the Intel 64 architecture, the
/// high-order 32 bits of each of RAX and RDX are cleared.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_rdtsc)
#[inline]
#[cfg_attr(test, assert_instr(rdtsc))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _rdtsc() -> u64 {
    rdtsc()
}

/// Reads the current value of the processor’s time-stamp counter and
/// the `IA32_TSC_AUX MSR`.
///
/// The processor monotonically increments the time-stamp counter MSR
/// every clock cycle and resets it to 0 whenever the processor is
/// reset.
///
/// The RDTSCP instruction waits until all previous instructions have
/// been executed before reading the counter. However, subsequent
/// instructions may begin execution before the read operation is
/// performed.
///
/// On processors that support the Intel 64 architecture, the
/// high-order 32 bits of each of RAX, RDX, and RCX are cleared.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=__rdtscp)
#[inline]
#[cfg_attr(test, assert_instr(rdtscp))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn __rdtscp(aux: *mut u32) -> u64 {
    let (tsc, auxval) = rdtscp();
    *aux = auxval;
    tsc
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.rdtsc"]
    fn rdtsc() -> u64;
    #[link_name = "llvm.x86.rdtscp"]
    fn rdtscp() -> (u64, u32);
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "sse2")]
    unsafe fn test_rdtsc() {
        let r = _rdtsc();
        assert_ne!(r, 0); // The chances of this being 0 are infinitesimal
    }

    #[simd_test(enable = "sse2")]
    unsafe fn test_rdtscp() {
        let mut aux = 0;
        let r = __rdtscp(&mut aux);
        assert_ne!(r, 0); // The chances of this being 0 are infinitesimal
    }
}
