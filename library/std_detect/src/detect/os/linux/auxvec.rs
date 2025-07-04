//! Parses ELF auxiliary vectors.
#![allow(dead_code)]

pub(crate) const AT_NULL: usize = 0;

/// Key to access the CPU Hardware capabilities bitfield.
pub(crate) const AT_HWCAP: usize = 16;
/// Key to access the CPU Hardware capabilities 2 bitfield.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "arm",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "s390x",
))]
pub(crate) const AT_HWCAP2: usize = 26;

/// Cache HWCAP bitfields of the ELF Auxiliary Vector.
///
/// If an entry cannot be read all the bits in the bitfield are set to zero.
/// This should be interpreted as all the features being disabled.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct AuxVec {
    pub hwcap: usize,
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "s390x",
    ))]
    pub hwcap2: usize,
}

/// ELF Auxiliary Vector
///
/// The auxiliary vector is a memory region in a running ELF program's stack
/// composed of (key: usize, value: usize) pairs.
///
/// The keys used in the aux vector are platform dependent. For Linux, they are
/// defined in [linux/auxvec.h][auxvec_h]. The hardware capabilities of a given
/// CPU can be queried with the  `AT_HWCAP` and `AT_HWCAP2` keys.
///
/// There is no perfect way of reading the auxiliary vector.
///
/// - If the `std_detect_dlsym_getauxval` cargo feature is enabled, this will use
///   `getauxval` if its linked to the binary, and otherwise proceed to a fallback implementation.
///   When `std_detect_dlsym_getauxval` is disabled, this will assume that `getauxval` is
///   linked to the binary - if that is not the case the behavior is undefined.
/// - Otherwise, if the `std_detect_file_io` cargo feature is enabled, it will
///   try to read `/proc/self/auxv`.
/// - If that fails, this function returns an error.
///
/// Note that run-time feature detection is not invoked for features that can
/// be detected at compile-time.
///
///  Note: The `std_detect_dlsym_getauxval` cargo feature is ignored on
/// `*-linux-{gnu,musl,ohos}*` and `*-android*` targets because we can safely assume `getauxval`
/// is linked to the binary.
/// - `*-linux-gnu*` targets ([since Rust 1.64](https://blog.rust-lang.org/2022/08/01/Increasing-glibc-kernel-requirements.html))
///   have glibc requirements higher than [glibc 2.16 that added `getauxval`](https://sourceware.org/legacy-ml/libc-announce/2012/msg00000.html).
/// - `*-linux-musl*` targets ([at least since Rust 1.15](https://github.com/rust-lang/rust/blob/1.15.0/src/ci/docker/x86_64-musl/build-musl.sh#L15))
///   use musl newer than [musl 1.1.0 that added `getauxval`](https://git.musl-libc.org/cgit/musl/tree/WHATSNEW?h=v1.1.0#n1197)
/// - `*-linux-ohos*` targets use a [fork of musl 1.2](https://gitee.com/openharmony/docs/blob/master/en/application-dev/reference/native-lib/musl.md)
/// - `*-android*` targets ([since Rust 1.68](https://blog.rust-lang.org/2023/01/09/android-ndk-update-r25.html))
///   have the minimum supported API level higher than [Android 4.3 (API level 18) that added `getauxval`](https://github.com/aosp-mirror/platform_bionic/blob/d3ebc2f7c49a9893b114124d4a6b315f3a328764/libc/include/sys/auxv.h#L49).
///
/// For more information about when `getauxval` is available check the great
/// [`auxv` crate documentation][auxv_docs].
///
/// [auxvec_h]: https://github.com/torvalds/linux/blob/master/include/uapi/linux/auxvec.h
/// [auxv_docs]: https://docs.rs/auxv/0.3.3/auxv/
pub(crate) fn auxv() -> Result<AuxVec, ()> {
    // Try to call a getauxval function.
    if let Ok(hwcap) = getauxval(AT_HWCAP) {
        // Targets with only AT_HWCAP:
        #[cfg(any(
            target_arch = "riscv32",
            target_arch = "riscv64",
            target_arch = "mips",
            target_arch = "mips64",
            target_arch = "loongarch64",
        ))]
        {
            // Zero could indicate that no features were detected, but it's also used to indicate
            // an error. In either case, try the fallback.
            if hwcap != 0 {
                return Ok(AuxVec { hwcap });
            }
        }

        // Targets with AT_HWCAP and AT_HWCAP2:
        #[cfg(any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "s390x",
        ))]
        {
            if let Ok(hwcap2) = getauxval(AT_HWCAP2) {
                // Zero could indicate that no features were detected, but it's also used to indicate
                // an error. In particular, on many platforms AT_HWCAP2 will be legitimately zero,
                // since it contains the most recent feature flags. Use the fallback only if no
                // features were detected at all.
                if hwcap != 0 || hwcap2 != 0 {
                    return Ok(AuxVec { hwcap, hwcap2 });
                }
            }
        }

        // Intentionnaly not used
        let _ = hwcap;
    }

    #[cfg(feature = "std_detect_file_io")]
    {
        // If calling getauxval fails, try to read the auxiliary vector from
        // its file:
        auxv_from_file("/proc/self/auxv")
    }
    #[cfg(not(feature = "std_detect_file_io"))]
    {
        Err(())
    }
}

/// Tries to read the `key` from the auxiliary vector by calling the
/// `getauxval` function. If the function is not linked, this function return `Err`.
fn getauxval(key: usize) -> Result<usize, ()> {
    type F = unsafe extern "C" fn(libc::c_ulong) -> libc::c_ulong;
    cfg_if::cfg_if! {
        if #[cfg(all(
            feature = "std_detect_dlsym_getauxval",
            not(all(
                target_os = "linux",
                any(target_env = "gnu", target_env = "musl", target_env = "ohos"),
            )),
            not(target_os = "android"),
        ))] {
            let ffi_getauxval: F = unsafe {
                let ptr = libc::dlsym(libc::RTLD_DEFAULT, c"getauxval".as_ptr());
                if ptr.is_null() {
                    return Err(());
                }
                core::mem::transmute(ptr)
            };
        } else {
            let ffi_getauxval: F = libc::getauxval;
        }
    }
    Ok(unsafe { ffi_getauxval(key as libc::c_ulong) as usize })
}

/// Tries to read the auxiliary vector from the `file`. If this fails, this
/// function returns `Err`.
#[cfg(feature = "std_detect_file_io")]
pub(super) fn auxv_from_file(file: &str) -> Result<AuxVec, ()> {
    let file = super::read_file(file)?;

    // See <https://github.com/torvalds/linux/blob/v5.15/include/uapi/linux/auxvec.h>.
    //
    // The auxiliary vector contains at most 34 (key,value) fields: from
    // `AT_MINSIGSTKSZ` to `AT_NULL`, but its number may increase.
    let len = file.len();
    let mut buf = alloc::vec![0_usize; 1 + len / core::mem::size_of::<usize>()];
    unsafe {
        core::ptr::copy_nonoverlapping(file.as_ptr(), buf.as_mut_ptr() as *mut u8, len);
    }

    auxv_from_buf(&buf)
}

/// Tries to interpret the `buffer` as an auxiliary vector. If that fails, this
/// function returns `Err`.
#[cfg(feature = "std_detect_file_io")]
fn auxv_from_buf(buf: &[usize]) -> Result<AuxVec, ()> {
    // Targets with only AT_HWCAP:
    #[cfg(any(
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "loongarch64",
    ))]
    {
        for el in buf.chunks(2) {
            match el[0] {
                AT_NULL => break,
                AT_HWCAP => return Ok(AuxVec { hwcap: el[1] }),
                _ => (),
            }
        }
    }
    // Targets with AT_HWCAP and AT_HWCAP2:
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "s390x",
    ))]
    {
        let mut hwcap = None;
        // For some platforms, AT_HWCAP2 was added recently, so let it default to zero.
        let mut hwcap2 = 0;
        for el in buf.chunks(2) {
            match el[0] {
                AT_NULL => break,
                AT_HWCAP => hwcap = Some(el[1]),
                AT_HWCAP2 => hwcap2 = el[1],
                _ => (),
            }
        }

        if let Some(hwcap) = hwcap {
            return Ok(AuxVec { hwcap, hwcap2 });
        }
    }
    // Suppress unused variable
    let _ = buf;
    Err(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // FIXME: on mips/mips64 getauxval returns 0, and /proc/self/auxv
    // does not always contain the AT_HWCAP key under qemu.
    #[cfg(any(
        target_arch = "arm",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "s390x",
    ))]
    #[test]
    fn auxv_crate() {
        let v = auxv();
        if let Ok(hwcap) = getauxval(AT_HWCAP) {
            let rt_hwcap = v.expect("failed to find hwcap key").hwcap;
            assert_eq!(rt_hwcap, hwcap);
        }

        // Targets with AT_HWCAP and AT_HWCAP2:
        #[cfg(any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "s390x",
        ))]
        {
            if let Ok(hwcap2) = getauxval(AT_HWCAP2) {
                let rt_hwcap2 = v.expect("failed to find hwcap2 key").hwcap2;
                assert_eq!(rt_hwcap2, hwcap2);
            }
        }
    }

    #[test]
    fn auxv_dump() {
        if let Ok(auxvec) = auxv() {
            println!("{:?}", auxvec);
        } else {
            println!("both getauxval() and reading /proc/self/auxv failed!");
        }
    }

    #[cfg(feature = "std_detect_file_io")]
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "arm")] {
            #[test]
            fn linux_rpi3() {
                let file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/linux-rpi3.auxv");
                println!("file: {file}");
                let v = auxv_from_file(file).unwrap();
                assert_eq!(v.hwcap, 4174038);
                assert_eq!(v.hwcap2, 16);
            }

            #[test]
            fn linux_macos_vb() {
                let file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/macos-virtualbox-linux-x86-4850HQ.auxv");
                println!("file: {file}");
                // The file contains HWCAP but not HWCAP2. In that case, we treat HWCAP2 as zero.
                let v = auxv_from_file(file).unwrap();
                assert_eq!(v.hwcap, 126614527);
                assert_eq!(v.hwcap2, 0);
            }
        } else if #[cfg(target_arch = "aarch64")] {
            #[cfg(target_endian = "little")]
            #[test]
            fn linux_artificial_aarch64() {
                let file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/linux-artificial-aarch64.auxv");
                println!("file: {file}");
                let v = auxv_from_file(file).unwrap();
                assert_eq!(v.hwcap, 0x0123456789abcdef);
                assert_eq!(v.hwcap2, 0x02468ace13579bdf);
            }
            #[cfg(target_endian = "little")]
            #[test]
            fn linux_no_hwcap2_aarch64() {
                let file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/linux-no-hwcap2-aarch64.auxv");
                println!("file: {file}");
                let v = auxv_from_file(file).unwrap();
                // An absent HWCAP2 is treated as zero, and does not prevent acceptance of HWCAP.
                assert_ne!(v.hwcap, 0);
                assert_eq!(v.hwcap2, 0);
            }
        }
    }

    #[test]
    #[cfg(feature = "std_detect_file_io")]
    fn auxv_dump_procfs() {
        if let Ok(auxvec) = auxv_from_file("/proc/self/auxv") {
            println!("{:?}", auxvec);
        } else {
            println!("reading /proc/self/auxv failed!");
        }
    }

    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "s390x",
    ))]
    #[test]
    #[cfg(feature = "std_detect_file_io")]
    fn auxv_crate_procfs() {
        if let Ok(procfs_auxv) = auxv_from_file("/proc/self/auxv") {
            assert_eq!(auxv().unwrap(), procfs_auxv);
        }
    }
}
