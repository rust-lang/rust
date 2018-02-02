//! Run-time feature detection for ARM and PowerPC64  on Linux.

use coresimd::__vendor_runtime::__runtime::cache;
use coresimd::__vendor_runtime::__runtime::arch;
pub use self::arch::__Feature;

#[cfg(target_arch = "arm")]
mod arm;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "powerpc64")]
mod powerpc64;

mod auxv;
mod cpuinfo;

/// Detects CPU features:
pub fn detect_features() -> cache::Initializer {
    // Try to read the ELF Auxiliary Vector using libc's getauxval:
    if let Ok(v) = auxv::libc::auxv() {
        return arch::detect_features(v);
    }
    // Try to read the ELF Auxiliary Vector from /proc/self/auxv:
    if let Ok(v) = auxv::proc_self::auxv() {
        return arch::detect_features(v);
    }
    // Try to read /proc/cpuinfo:
    if let Ok(v) = cpuinfo::CpuInfo::new() {
        return arch::detect_features(v);
    }
    // Otherwise all features are disabled
    cache::Initializer::default()
}

/// Performs run-time feature detection.
pub fn __unstable_detect_feature(x: __Feature) -> bool {
    cache::test(x as u32, detect_features)
}
