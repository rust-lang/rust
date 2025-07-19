//@ run-pass
#![allow(unused_variables)]
#![allow(stable_features)]
#![allow(overflowing_literals)]

//@ needs-subprocess
//@ ignore-fuchsia must translate zircon signal to SIGILL, FIXME (#58590)

#![feature(repr_simd, target_feature, cfg_target_feature)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
#[allow(unused)]
use minisimd::*;

use std::process::{Command, ExitStatus};
use std::env;

fn main() {
    if let Some(level) = env::args().nth(1) {
        return test::main(&level)
    }

    let me = env::current_exe().unwrap();
    for level in ["sse", "avx", "avx512"].iter() {
        let status = Command::new(&me).arg(level).status().unwrap();
        if status.success() {
            println!("success with {}", level);
            continue
        }

        // We don't actually know if our computer has the requisite target features
        // for the test below. Testing for that will get added to libstd later so
        // for now just assume sigill means this is a machine that can't run this test.
        if is_sigill(status) {
            println!("sigill with {}, assuming spurious", level);
            continue
        }
        panic!("invalid status at {}: {}", level, status);
    }
}

#[cfg(unix)]
fn is_sigill(status: ExitStatus) -> bool {
    use std::os::unix::prelude::*;
    status.signal() == Some(4)
}

#[cfg(windows)]
fn is_sigill(status: ExitStatus) -> bool {
    status.code() == Some(0xc000001d)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(nonstandard_style)]
mod test {
    // An SSE type
    type __m128i = super::u64x2;

    // An AVX type
    type __m256i = super::u64x4;

    // An AVX-512 type
    type __m512i = super::u64x8;

    pub fn main(level: &str) {
        unsafe {
            main_normal(level);
            main_sse(level);
            if level == "sse" {
                return
            }
            main_avx(level);
            if level == "avx" {
                return
            }
            main_avx512(level);
        }
    }

    macro_rules! mains {
        ($(
            $(#[$attr:meta])*
            unsafe fn $main:ident(level: &str) {
                ...
            }
        )*) => ($(
            $(#[$attr])*
            unsafe fn $main(level: &str) {
                let m128 = __m128i::from_array([1, 2]);
                let m256 = __m256i::from_array([3, 4, 5, 6]);
                let m512 = __m512i::from_array([7, 8, 9, 10, 11, 12, 13, 14]);
                assert_eq!(id_sse_128(m128), m128);
                assert_eq!(id_sse_256(m256), m256);
                assert_eq!(id_sse_512(m512), m512);

                if level == "sse" {
                    return
                }
                assert_eq!(id_avx_128(m128), m128);
                assert_eq!(id_avx_256(m256), m256);
                assert_eq!(id_avx_512(m512), m512);

                if level == "avx" {
                    return
                }
                assert_eq!(id_avx512_128(m128), m128);
                assert_eq!(id_avx512_256(m256), m256);
                assert_eq!(id_avx512_512(m512), m512);
            }
        )*)
    }

    mains! {
        unsafe fn main_normal(level: &str) { ... }
        #[target_feature(enable = "sse2")]
        unsafe fn main_sse(level: &str) { ... }
        #[target_feature(enable = "avx")]
        unsafe fn main_avx(level: &str) { ... }
        #[target_feature(enable = "avx512bw")]
        unsafe fn main_avx512(level: &str) { ... }
    }


    #[target_feature(enable = "sse2")]
    unsafe fn id_sse_128(a: __m128i) -> __m128i {
        assert_eq!(a, __m128i::from_array([1, 2]));
        a.clone()
    }

    #[target_feature(enable = "sse2")]
    unsafe fn id_sse_256(a: __m256i) -> __m256i {
        assert_eq!(a, __m256i::from_array([3, 4, 5, 6]));
        a.clone()
    }

    #[target_feature(enable = "sse2")]
    unsafe fn id_sse_512(a: __m512i) -> __m512i {
        assert_eq!(a, __m512i::from_array([7, 8, 9, 10, 11, 12, 13, 14]));
        a.clone()
    }

    #[target_feature(enable = "avx")]
    unsafe fn id_avx_128(a: __m128i) -> __m128i {
        assert_eq!(a, __m128i::from_array([1, 2]));
        a.clone()
    }

    #[target_feature(enable = "avx")]
    unsafe fn id_avx_256(a: __m256i) -> __m256i {
        assert_eq!(a, __m256i::from_array([3, 4, 5, 6]));
        a.clone()
    }

    #[target_feature(enable = "avx")]
    unsafe fn id_avx_512(a: __m512i) -> __m512i {
        assert_eq!(a, __m512i::from_array([7, 8, 9, 10, 11, 12, 13, 14]));
        a.clone()
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn id_avx512_128(a: __m128i) -> __m128i {
        assert_eq!(a, __m128i::from_array([1, 2]));
        a.clone()
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn id_avx512_256(a: __m256i) -> __m256i {
        assert_eq!(a, __m256i::from_array([3, 4, 5, 6]));
        a.clone()
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn id_avx512_512(a: __m512i) -> __m512i {
        assert_eq!(a, __m512i::from_array([7, 8, 9, 10, 11, 12, 13, 14]));
        a.clone()
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod test {
    pub fn main(level: &str) {}
}
