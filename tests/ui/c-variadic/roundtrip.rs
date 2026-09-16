//@ run-pass
//@ ignore-backends: gcc
// tidy-alphabetical-start
#![feature(c_variadic_int128)]
#![feature(c_variadic_va_arg_safe)]
#![feature(const_c_variadic)]
#![feature(const_cmp)]
#![feature(const_convert)]
#![feature(const_destruct)]
#![feature(const_option_ops)]
#![feature(const_raw_ptr_comparison)]
#![feature(const_trait_impl)]
// tidy-alphabetical-end
#![allow(static_mut_refs)]
#![allow(unused_features)] // c_variadic_int128 is only used on 64-bit targets.

use std::convert::identity;
use std::ffi::*;
use std::num::NonZero;
use std::ptr::{self, NonNull};

// In rustc we implement `va_arg` for the callee reading from a VaList, but still rely on LLVM
// for exactly how to pass c-variadic arguments and for constructing the VaList. Here we test
// that the rustc implementation works with what LLVM gives us.

#[allow(improper_ctypes_definitions)]
const unsafe extern "C" fn variadic<T: VaArgSafe>(mut ap: ...) -> (T, T) {
    let x = ap.next_arg::<T>();
    // Intersperse a small type to test alignment logic. A `u32` (i.e. `c_uint`) is the smallest
    // type that implements `VaArgSafe` (except on some 16-bit targets): smaller types would
    // automatically be promoted.
    assert!(ap.next_arg::<u32>() == 0xAAAA_AAAA);
    let y = ap.next_arg::<T>();

    (x, y)
}

macro_rules! roundtrip {
    ($ty:ty, $a:expr, $b:expr) => {
        const {
            let a: $ty = $a;
            let b: $ty = $b;
            let (x, y) = variadic::<$ty>(a, 0xAAAA_AAAAu32, b);
            assert!(a == x);
            assert!(b == y);
        }

        let a: $ty = $a;
        let b: $ty = $b;
        assert_eq!(variadic::<$ty>(a, 0xAAAA_AAAAu32, b), (a, b))
    };
}

macro_rules! roundtrip_zeroable {
    ($ty:ty, $a:expr, $b:expr) => {
        roundtrip!($ty, $a, $b);
        roundtrip!(NonZero<$ty>, NonZero::new($a).unwrap(), NonZero::new($b).unwrap());
        roundtrip!(Option<NonZero<$ty>>, None, NonZero::new($a));
    };
}

macro_rules! roundtrip_ptr {
    ($ty:ty, $a:expr, $b:expr) => {
        roundtrip_ptr!($ty, $a, $b, identity);
    };
    ($ty:ty, $a:expr, $b:expr, $conv:expr) => {
        const {
            let a: $ty = $a;
            let b: $ty = $b;
            let conv = $conv;

            // note: we need to convert references to raw pointers to avoid aliasing
            let a2 = conv(a);
            let b2 = conv(b);
            let (x, y) = variadic::<$ty>(a, 0xAAAA_AAAAu32, b);
            assert!(a2.guaranteed_eq(conv(x)).unwrap());
            assert!(b2.guaranteed_eq(conv(y)).unwrap());
        }

        let a: $ty = $a;
        let b: $ty = $b;
        let conv = $conv;

        // note: similarly here, we can't alias the original pointers
        let a2 = conv(a);
        let b2 = conv(b);
        let (x, y) = variadic::<$ty>(a, 0xAAAA_AAAAu32, b);
        assert_eq!((conv(x), conv(y)), (a2, b2))
    };
}

fn main() {
    unsafe {
        roundtrip_zeroable!(i32, -1, -2);
        roundtrip_zeroable!(i64, -1, -2);
        roundtrip_zeroable!(isize, -1, -2);
        roundtrip_zeroable!(c_int, -1, -2);
        roundtrip_zeroable!(c_long, -1, -2);
        roundtrip_zeroable!(c_longlong, -1, -2);

        roundtrip_zeroable!(u32, 1, 2);
        roundtrip_zeroable!(u64, 1, 2);
        roundtrip_zeroable!(usize, 1, 2);
        roundtrip_zeroable!(c_uint, 1, 2);
        roundtrip_zeroable!(c_ulong, 1, 2);
        roundtrip_zeroable!(c_ulonglong, 1, 2);

        roundtrip!(f64, 3.14, 6.28);
        roundtrip!(c_double, 3.14, 6.28);

        static mut A: u32 = 1u32;
        static mut B: u32 = 2u32;
        roundtrip_ptr!(*const u32, &raw const A, &raw const B);
        roundtrip_ptr!(*mut u32, &raw mut A, &raw mut B);
        roundtrip_ptr!(&u32, &A, &B, ptr::from_ref);
        roundtrip_ptr!(&mut u32, &mut A, &mut B, ptr::from_mut);
        roundtrip_ptr!(NonNull<u32>, NonNull::from(&A), NonNull::from(&B), NonNull::as_ptr);
        const fn option_as_ptr(x: Option<NonNull<u32>>) -> *mut u32 {
            x.map_or_else(ptr::null_mut, NonNull::as_ptr)
        }
        roundtrip_ptr!(
            Option<NonNull<u32>>,
            NonNull::new(&raw mut A),
            NonNull::new(&raw mut B),
            option_as_ptr
        );

        // The 128-bit integers only implement VaArgSafe on some targets, a subset of those that
        // define `__int128`. We test some of those targets here.
        cfg_select! {
            any(
                target_arch = "aarch64",
                target_arch = "amdgpu",
                target_arch = "arm64ec",
                target_arch = "bpf",
                target_arch = "loongarch64",
                target_arch = "mips64",
                target_arch = "mips64r6",
                target_arch = "nvptx64",
                target_arch = "powerpc64",
                target_arch = "riscv64",
                target_arch = "s390x",
                target_arch = "sparc64",
                target_arch = "wasm32",
                target_arch = "wasm64",
                target_arch = "x86_64",
            ) => {
                #[cfg(not(any(
                    target_arch = "wasm32",
                    target_abi = "x32",
                    target_pointer_width = "64",
                )))]
                compile_error!("unexpected target architecture for 128-bit c-variadic");

                roundtrip_zeroable!(i128, -1, -2);
                roundtrip_zeroable!(u128, 1, 2);
            }
            _ => {}
        }
    }
}
