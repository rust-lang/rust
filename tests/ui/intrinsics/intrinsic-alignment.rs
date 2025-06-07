//@ run-pass

#![feature(core_intrinsics, rustc_attrs)]

use std::intrinsics as rusti;

#[cfg(any(
    target_os = "aix",
    target_os = "android",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "fuchsia",
    target_os = "hurd",
    target_os = "illumos",
    target_os = "linux",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "solaris",
    target_os = "vxworks",
    target_os = "nto",
    target_vendor = "apple",
))]
mod m {
    #[cfg(target_arch = "x86")]
    pub fn main() {
        assert_eq!(crate::rusti::min_align_of::<u64>(), 4);
    }

    #[cfg(not(target_arch = "x86"))]
    pub fn main() {
        assert_eq!(crate::rusti::min_align_of::<u64>(), 8);
    }
}

#[cfg(target_env = "sgx")]
mod m {
    #[cfg(target_arch = "x86_64")]
    pub fn main() {
        assert_eq!(crate::rusti::min_align_of::<u64>(), 8);
    }
}

#[cfg(target_os = "windows")]
mod m {
    pub fn main() {
        assert_eq!(crate::rusti::min_align_of::<u64>(), 8);
    }
}

#[cfg(target_family = "wasm")]
mod m {
    pub fn main() {
        assert_eq!(crate::rusti::min_align_of::<u64>(), 8);
    }
}

fn main() {
    m::main();
}
