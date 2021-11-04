// run-pass
#![allow(dead_code)]
#![allow(unused_unsafe)]
// ignore-wasm32-bare seems unimportant to test

// Issue #2303

#![feature(intrinsics)]

use std::mem;

mod rusti {
    extern "rust-intrinsic" {
        pub fn pref_align_of<T>() -> usize;
        pub fn min_align_of<T>() -> usize;
    }
}

// This is the type with the questionable alignment
#[derive(Debug)]
struct Inner {
    c64: u64
}

// This is the type that contains the type with the
// questionable alignment, for testing
#[derive(Debug)]
struct Outer {
    c8: u8,
    t: Inner
}


#[cfg(any(target_os = "android",
          target_os = "dragonfly",
          target_os = "emscripten",
          target_os = "freebsd",
          target_os = "fuchsia",
          target_os = "illumos",
          target_os = "linux",
          target_os = "macos",
          target_os = "netbsd",
          target_os = "openbsd",
          target_os = "solaris",
          target_os = "vxworks"))]
mod m {
    #[cfg(target_arch = "x86")]
    pub mod m {
        pub fn align() -> usize { 4 }
        pub fn size() -> usize { 12 }
    }

    #[cfg(not(target_arch = "x86"))]
    pub mod m {
        pub fn align() -> usize { 8 }
        pub fn size() -> usize { 16 }
    }
}

#[cfg(target_env = "sgx")]
mod m {
    #[cfg(target_arch = "x86_64")]
    pub mod m {
        pub fn align() -> usize { 8 }
        pub fn size() -> usize { 16 }
    }
}

#[cfg(target_os = "windows")]
mod m {
    pub mod m {
        pub fn align() -> usize { 8 }
        pub fn size() -> usize { 16 }
    }
}

pub fn main() {
    unsafe {
        let x = Outer {c8: 22, t: Inner {c64: 44}};

        let y = format!("{:?}", x);

        println!("align inner = {:?}", rusti::min_align_of::<Inner>());
        println!("size outer = {:?}", mem::size_of::<Outer>());
        println!("y = {:?}", y);

        // per clang/gcc the alignment of `Inner` is 4 on x86.
        assert_eq!(rusti::min_align_of::<Inner>(), m::m::align());

        // per clang/gcc the size of `Outer` should be 12
        // because `Inner`s alignment was 4.
        assert_eq!(mem::size_of::<Outer>(), m::m::size());

        assert_eq!(y, "Outer { c8: 22, t: Inner { c64: 44 } }".to_string());
    }
}
