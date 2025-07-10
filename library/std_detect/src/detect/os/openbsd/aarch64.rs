//! Run-time feature detection for Aarch64 on OpenBSD.
//!
//! OpenBSD doesn't trap the mrs instruction, but exposes the system registers through sysctl.
//! https://github.com/openbsd/src/commit/d335af936b9d7dd9cf655cae1ce19560c45de6c8
//! https://github.com/golang/go/commit/cd54ef1f61945459486e9eea2f016d99ef1da925

use core::mem::MaybeUninit;
use core::ptr;

use crate::detect::cache;

// Defined in machine/cpu.h.
// https://github.com/openbsd/src/blob/72ccc03bd11da614f31f7ff76e3f6fce99bc1c79/sys/arch/arm64/include/cpu.h#L25-L40
const CPU_ID_AA64ISAR0: libc::c_int = 2;
const CPU_ID_AA64ISAR1: libc::c_int = 3;
const CPU_ID_AA64MMFR2: libc::c_int = 7;
const CPU_ID_AA64PFR0: libc::c_int = 8;

/// Try to read the features from the system registers.
pub(crate) fn detect_features() -> cache::Initializer {
    // ID_AA64ISAR0_EL1 and ID_AA64ISAR1_EL1 are supported on OpenBSD 7.1+.
    // https://github.com/openbsd/src/commit/d335af936b9d7dd9cf655cae1ce19560c45de6c8
    // Others are supported on OpenBSD 7.3+.
    // https://github.com/openbsd/src/commit/c7654cd65262d532212f65123ee3905ba200365c
    // sysctl returns an unsupported error if operation is not supported,
    // so we can safely use this function on older versions of OpenBSD.
    let aa64isar0 = sysctl64(&[libc::CTL_MACHDEP, CPU_ID_AA64ISAR0]).unwrap_or(0);
    let aa64isar1 = sysctl64(&[libc::CTL_MACHDEP, CPU_ID_AA64ISAR1]).unwrap_or(0);
    let aa64mmfr2 = sysctl64(&[libc::CTL_MACHDEP, CPU_ID_AA64MMFR2]).unwrap_or(0);
    // Do not use unwrap_or(0) because in fp and asimd fields, 0 indicates that
    // the feature is available.
    let aa64pfr0 = sysctl64(&[libc::CTL_MACHDEP, CPU_ID_AA64PFR0]);

    super::aarch64::parse_system_registers(aa64isar0, aa64isar1, aa64mmfr2, aa64pfr0)
}

#[inline]
fn sysctl64(mib: &[libc::c_int]) -> Option<u64> {
    const OUT_LEN: libc::size_t = core::mem::size_of::<u64>();
    let mut out = MaybeUninit::<u64>::uninit();
    let mut out_len = OUT_LEN;
    let res = unsafe {
        libc::sysctl(
            mib.as_ptr(),
            mib.len() as libc::c_uint,
            out.as_mut_ptr() as *mut libc::c_void,
            &mut out_len,
            ptr::null_mut(),
            0,
        )
    };
    if res == -1 || out_len != OUT_LEN {
        return None;
    }
    // SAFETY: we've checked that sysctl was successful and `out` was filled.
    Some(unsafe { out.assume_init() })
}
