//! ELF Auxiliary Vector
//!
//! The auxiliary vector is a memory region in a running ELF program's stack
//! composed of (key: usize, value: usize) pairs.
//!
//! The keys used in the aux vector are platform dependent. For Linux, they are
//! defined in [linux/auxvec.h][auxvec_h]. The hardware capabilities of a given
//! CPU can be queried with the  `AT_HWCAP` and `AT_HWCAP2` keys.
//!
//! There is no perfect way of reading the auxiliary vector.
//!
//! - `coresimd`: if `getauxval` is available, `coresimd` will try to use it.
//! - `stdsimd`: if `getauxval` is not available, it will try to read
//! `/proc/self/auxv`, and if that fails it will try to read `/proc/cpuinfo`.
//!
//! For more information about when `getauxval` is available check the great
//! [`auxv` crate documentation][auxv_docs].
//!
//! [auxvec_h]: https://github.com/torvalds/linux/blob/master/include/uapi/linux/auxvec.h
//! [auxv_docs]: https://docs.rs/auxv/0.3.3/auxv/

/// Key to access the CPU Hardware capabilities bitfield.
pub const AT_HWCAP: usize = 16;
/// Key to access the CPU Hardware capabilities 2 bitfield.
pub const AT_HWCAP2: usize = 26;

/// Cache HWCAP bitfields of the ELF Auxiliary Vector.
///
/// If an entry cannot be read all the bits in the bitfield
/// are set to zero.
#[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
#[derive(Debug, Copy, Clone)]
pub struct AuxVec {
    pub hwcap: usize,
    pub hwcap2: usize,
}

/// Cache HWCAP bitfields of the ELF Auxiliary Vector.
///
/// If an entry cannot be read all the bits in the bitfield
/// are set to zero.
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Copy, Clone)]
pub struct AuxVec {
    pub hwcap: usize,
}
