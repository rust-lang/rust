//! Parses ELF auxiliary vectors.
#![cfg_attr(
    any(target_arch = "aarch64", target_arch = "powerpc64", target_arch = "riscv64"),
    allow(dead_code)
)]

/// Cache HWCAP bitfields of the ELF Auxiliary Vector.
///
/// If an entry cannot be read all the bits in the bitfield are set to zero.
/// This should be interpreted as all the features being disabled.
#[derive(Debug, Copy, Clone)]
pub(crate) struct AuxVec {
    pub hwcap: usize,
    pub hwcap2: usize,
}

/// ELF Auxiliary Vector
///
/// The auxiliary vector is a memory region in a running ELF program's stack
/// composed of (key: usize, value: usize) pairs.
///
/// The keys used in the aux vector are platform dependent. For OpenBSD, they are
/// defined in [machine/elf.h][elfh]. The hardware capabilities of a given CPU
/// can be queried with the  `AT_HWCAP` and `AT_HWCAP2` keys.
///
/// Note that run-time feature detection is not invoked for features that can
/// be detected at compile-time.
///
/// [elf.h]: https://github.com/openbsd/src/blob/master/sys/arch/arm64/include/elf.h
/// [elf.h]: https://github.com/openbsd/src/blob/master/sys/arch/powerpc64/include/elf.h
pub(crate) fn auxv() -> Result<AuxVec, ()> {
    let hwcap = archauxv(libc::AT_HWCAP);
    let hwcap2 = archauxv(libc::AT_HWCAP2);
    // Zero could indicate that no features were detected, but it's also used to
    // indicate an error. In particular, on many platforms AT_HWCAP2 will be
    // legitimately zero, since it contains the most recent feature flags.
    if hwcap != 0 || hwcap2 != 0 {
        return Ok(AuxVec { hwcap, hwcap2 });
    }
    Err(())
}

/// Tries to read the `key` from the auxiliary vector.
fn archauxv(key: libc::c_int) -> usize {
    const OUT_LEN: libc::c_int = core::mem::size_of::<libc::c_ulong>() as libc::c_int;
    let mut out: libc::c_ulong = 0;
    unsafe {
        let res =
            libc::elf_aux_info(key, &mut out as *mut libc::c_ulong as *mut libc::c_void, OUT_LEN);
        // If elf_aux_info fails, `out` will be left at zero (which is the proper default value).
        debug_assert!(res == 0 || out == 0);
    }
    out as usize
}
