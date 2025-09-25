//! Run-time feature detection on Linux
//!
#[cfg(feature = "std_detect_file_io")]
use alloc::vec::Vec;

mod auxvec;

#[cfg(feature = "std_detect_file_io")]
fn read_file(orig_path: &str) -> Result<Vec<u8>, alloc::string::String> {
    use alloc::format;

    let mut path = Vec::from(orig_path.as_bytes());
    path.push(0);

    unsafe {
        let file = libc::open(path.as_ptr() as *const libc::c_char, libc::O_RDONLY);
        if file == -1 {
            return Err(format!("Cannot open file at {orig_path}"));
        }

        let mut data = Vec::new();
        loop {
            data.reserve(4096);
            let spare = data.spare_capacity_mut();
            match libc::read(file, spare.as_mut_ptr() as *mut _, spare.len()) {
                -1 => {
                    libc::close(file);
                    return Err(format!("Error while reading from file at {orig_path}"));
                }
                0 => break,
                n => data.set_len(data.len() + n as usize),
            }
        }

        libc::close(file);
        Ok(data)
    }
}

cfg_select! {
    target_arch = "aarch64" => {
        mod aarch64;
        pub(crate) use self::aarch64::detect_features;
    }
    target_arch = "arm" => {
        mod arm;
        pub(crate) use self::arm::detect_features;
    }
    any(target_arch = "riscv32", target_arch = "riscv64") => {
        mod riscv;
        pub(crate) use self::riscv::detect_features;
    }
    any(target_arch = "mips", target_arch = "mips64") => {
        mod mips;
        pub(crate) use self::mips::detect_features;
    }
    any(target_arch = "powerpc", target_arch = "powerpc64") => {
        mod powerpc;
        pub(crate) use self::powerpc::detect_features;
    }
    any(target_arch = "loongarch32", target_arch = "loongarch64") => {
        mod loongarch;
        pub(crate) use self::loongarch::detect_features;
    }
    target_arch = "s390x" => {
        mod s390x;
        pub(crate) use self::s390x::detect_features;
    }
    _ => {
        use crate::detect::cache;
        /// Performs run-time feature detection.
        pub(crate) fn detect_features() -> cache::Initializer {
            cache::Initializer::default()
        }
    }
}
