//! Parses ELF auxiliary vectors.
#![cfg_attr(any(target_arch = "arm", target_arch = "powerpc64"), allow(dead_code))]

/// Key to access the CPU Hardware capabilities bitfield.
pub(crate) const AT_HWCAP: usize = 25;
/// Key to access the CPU Hardware capabilities 2 bitfield.
pub(crate) const AT_HWCAP2: usize = 26;

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
/// The keys used in the aux vector are platform dependent. For FreeBSD, they are
/// defined in [sys/elf_common.h][elf_common_h]. The hardware capabilities of a given
/// CPU can be queried with the  `AT_HWCAP` and `AT_HWCAP2` keys.
///
/// Note that run-time feature detection is not invoked for features that can
/// be detected at compile-time.
///
/// [elf_common.h]: https://svnweb.freebsd.org/base/release/12.0.0/sys/sys/elf_common.h?revision=341707
pub(crate) fn auxv() -> Result<AuxVec, ()> {
    if let Ok(hwcap) = archauxv(AT_HWCAP) {
        if let Ok(hwcap2) = archauxv(AT_HWCAP2) {
            if hwcap != 0 && hwcap2 != 0 {
                return Ok(AuxVec { hwcap, hwcap2 });
            }
        }
    }
    Err(())
}

/// Tries to read the `key` from the auxiliary vector.
fn archauxv(key: usize) -> Result<usize, ()> {
    use core::mem;

    #[derive(Copy, Clone)]
    #[repr(C)]
    pub struct Elf_Auxinfo {
        pub a_type: usize,
        pub a_un: unnamed,
    }
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub union unnamed {
        pub a_val: libc::c_long,
        pub a_ptr: *mut libc::c_void,
        pub a_fcn: Option<unsafe extern "C" fn() -> ()>,
    }

    let mut auxv: [Elf_Auxinfo; 27] = [Elf_Auxinfo {
        a_type: 0,
        a_un: unnamed { a_val: 0 },
    }; 27];

    let mut len: libc::c_uint = mem::size_of_val(&auxv) as libc::c_uint;

    unsafe {
        let mut mib = [
            libc::CTL_KERN,
            libc::KERN_PROC,
            libc::KERN_PROC_AUXV,
            libc::getpid(),
        ];

        let ret = libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as u32,
            &mut auxv as *mut _ as *mut _,
            &mut len as *mut _ as *mut _,
            0 as *mut libc::c_void,
            0,
        );

        if ret != -1 {
            for i in 0..auxv.len() {
                if auxv[i].a_type == key {
                    return Ok(auxv[i].a_un.a_val as usize);
                }
            }
        }
    }
    return Ok(0);
}
