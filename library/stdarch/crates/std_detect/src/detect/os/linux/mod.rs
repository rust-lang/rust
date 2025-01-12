//! Run-time feature detection on Linux
//!
#[cfg(feature = "std_detect_file_io")]
use alloc::vec::Vec;

mod auxvec;

#[cfg(feature = "std_detect_file_io")]
mod cpuinfo;

#[cfg(feature = "std_detect_file_io")]
fn read_file(path: &str) -> Result<Vec<u8>, ()> {
    let mut path = Vec::from(path.as_bytes());
    path.push(0);

    unsafe {
        let file = libc::open(path.as_ptr() as *const libc::c_char, libc::O_RDONLY);
        if file == -1 {
            return Err(());
        }

        let mut data = Vec::new();
        loop {
            data.reserve(4096);
            let spare = data.spare_capacity_mut();
            match libc::read(file, spare.as_mut_ptr() as *mut _, spare.len()) {
                -1 => {
                    libc::close(file);
                    return Err(());
                }
                0 => break,
                n => data.set_len(data.len() + n as usize),
            }
        }

        libc::close(file);
        Ok(data)
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_arch = "aarch64")] {
        mod aarch64;
        pub(crate) use self::aarch64::detect_features;
    } else if #[cfg(target_arch = "arm")] {
        mod arm;
        pub(crate) use self::arm::detect_features;
    } else if #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))] {
        mod riscv;
        pub(crate) use self::riscv::detect_features;
    } else if #[cfg(any(target_arch = "mips", target_arch = "mips64"))] {
        mod mips;
        pub(crate) use self::mips::detect_features;
    } else if #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))] {
        mod powerpc;
        pub(crate) use self::powerpc::detect_features;
    } else if #[cfg(target_arch = "loongarch64")] {
        mod loongarch;
        pub(crate) use self::loongarch::detect_features;
    } else if #[cfg(target_arch = "s390x")] {
        mod s390x;
        pub(crate) use self::s390x::detect_features;
    } else {
        use crate::detect::cache;
        /// Performs run-time feature detection.
        pub(crate) fn detect_features() -> cache::Initializer {
            cache::Initializer::default()
        }
    }
}
