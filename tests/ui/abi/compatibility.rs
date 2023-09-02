// check-pass
#![feature(rustc_attrs)]
#![allow(unused, improper_ctypes_definitions)]
use std::num::NonZeroI32;
use std::ptr::NonNull;

macro_rules! assert_abi_compatible {
    ($name:ident, $t1:ty, $t2:ty) => {
        mod $name {
            use super::*;
            // Test argument and return value, `Rust` and `C` ABIs.
            #[rustc_abi(assert_eq)]
            type TestRust = (fn($t1) -> $t1, fn($t2) -> $t2);
            #[rustc_abi(assert_eq)]
            type TestC = (extern "C" fn($t1) -> $t1, extern "C" fn($t2) -> $t2);
        }
    };
}

#[derive(Copy, Clone)]
struct Zst;

#[repr(C)]
struct ReprC1<T>(T);
#[repr(C)]
struct ReprC2Int<T>(i32, T);
#[repr(C)]
struct ReprC2Float<T>(f32, T);
#[repr(C)]
struct ReprC4<T>(T, T, T, T);
#[repr(C)]
struct ReprC4Mixed<T>(T, f32, i32, T);
#[repr(C)]
enum ReprCEnum<T> {
    Variant1,
    Variant2(T),
}
#[repr(C)]
union ReprCUnion<T: Copy> {
    nothing: (),
    something: T,
}

macro_rules! test_abi_compatible {
    ($name:ident, $t1:ty, $t2:ty) => {
        mod $name {
            use super::*;
            assert_abi_compatible!(plain, $t1, $t2);
            // We also do some tests with differences in fields of `repr(C)` types.
            assert_abi_compatible!(repr_c_1, ReprC1<$t1>, ReprC1<$t2>);
            assert_abi_compatible!(repr_c_2_int, ReprC2Int<$t1>, ReprC2Int<$t2>);
            assert_abi_compatible!(repr_c_2_float, ReprC2Float<$t1>, ReprC2Float<$t2>);
            assert_abi_compatible!(repr_c_4, ReprC4<$t1>, ReprC4<$t2>);
            assert_abi_compatible!(repr_c_4mixed, ReprC4Mixed<$t1>, ReprC4Mixed<$t2>);
            assert_abi_compatible!(repr_c_enum, ReprCEnum<$t1>, ReprCEnum<$t2>);
            assert_abi_compatible!(repr_c_union, ReprCUnion<$t1>, ReprCUnion<$t2>);
        }
    };
}

// Compatibility of pointers is probably de-facto guaranteed,
// but that does not seem to be documented.
test_abi_compatible!(ptr_mut, *const i32, *mut i32);
test_abi_compatible!(ptr_pointee, *const i32, *const Vec<i32>);
test_abi_compatible!(ref_mut, &i32, &mut i32);
test_abi_compatible!(ref_ptr, &i32, *const i32);
test_abi_compatible!(box_ptr, Box<i32>, *const i32);
test_abi_compatible!(nonnull_ptr, NonNull<i32>, *const i32);
test_abi_compatible!(fn_fn, fn(), fn(i32) -> i32);

// Some further guarantees we will likely (have to) make.
test_abi_compatible!(zst_unit, Zst, ());
test_abi_compatible!(zst_array, Zst, [u8; 0]);
test_abi_compatible!(nonzero_int, NonZeroI32, i32);

fn main() {}
