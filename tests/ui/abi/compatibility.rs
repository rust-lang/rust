// check-pass
#![feature(rustc_attrs, unsized_fn_params, transparent_unions)]
#![allow(unused, improper_ctypes_definitions, internal_features)]
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::NonZeroI32;
use std::ptr::NonNull;

// FIXME: a bunch of targets are broken in various ways.
// Hence there are `cfg` throughout this test to disable parts of it on those targets.
// sparc64: https://github.com/rust-lang/rust/issues/115336
// mips64: https://github.com/rust-lang/rust/issues/115404

macro_rules! assert_abi_compatible {
    ($name:ident, $t1:ty, $t2:ty) => {
        mod $name {
            use super::*;
            // Declaring a `type` doesn't even check well-formedness, so we also declare a function.
            fn check_wf(_x: $t1, _y: $t2) {}
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
struct ReprC1<T: ?Sized>(T);
#[repr(C)]
struct ReprC2Int<T>(i32, T);
#[repr(C)]
struct ReprC2Float<T>(f32, T);
#[repr(C)]
struct ReprC4<T>(T, Vec<i32>, Zst, T);
#[repr(C)]
struct ReprC4Mixed<T>(T, f32, i32, T);
#[repr(C)]
enum ReprCEnum<T> {
    Variant1,
    Variant2(T),
}
#[repr(C)]
union ReprCUnion<T> {
    nothing: (),
    something: ManuallyDrop<T>,
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
#[cfg(not(any(target_arch = "sparc64")))]
test_abi_compatible!(zst_array, Zst, [u8; 0]);
test_abi_compatible!(nonzero_int, NonZeroI32, i32);

// `DispatchFromDyn` relies on ABI compatibility.
// This is interesting since these types are not `repr(transparent)`.
test_abi_compatible!(rc, std::rc::Rc<i32>, *mut i32);
test_abi_compatible!(arc, std::sync::Arc<i32>, *mut i32);

// `repr(transparent)` compatibility.
#[repr(transparent)]
struct Wrapper1<T: ?Sized>(T);
#[repr(transparent)]
struct Wrapper2<T: ?Sized>((), Zst, T);
#[repr(transparent)]
struct Wrapper3<T>(T, [u8; 0], PhantomData<u64>);
#[repr(transparent)]
union WrapperUnion<T> {
    nothing: (),
    something: ManuallyDrop<T>,
}

macro_rules! test_transparent {
    ($name:ident, $t:ty) => {
        mod $name {
            use super::*;
            test_abi_compatible!(wrap1, $t, Wrapper1<$t>);
            test_abi_compatible!(wrap2, $t, Wrapper2<$t>);
            test_abi_compatible!(wrap3, $t, Wrapper3<$t>);
            test_abi_compatible!(wrap4, $t, WrapperUnion<$t>);
        }
    };
}

test_transparent!(simple, i32);
test_transparent!(reference, &'static i32);
test_transparent!(zst, Zst);
test_transparent!(unit, ());
test_transparent!(enum_, Option<i32>);
test_transparent!(enum_niched, Option<&'static i32>);
#[cfg(not(any(target_arch = "mips64", target_arch = "sparc64")))]
mod tuples {
    use super::*;
    // mixing in some floats since they often get special treatment
    test_transparent!(pair, (i32, f32));
    // chosen to fit into 64bit
    test_transparent!(triple, (i8, i16, f32));
    // Pure-float types that are not ScalarPair seem to be tricky.
    test_transparent!(triple_f32, (f32, f32, f32));
    test_transparent!(triple_f64, (f64, f64, f64));
    // and also something that's larger than 2 pointers
    test_transparent!(tuple, (i32, f32, i64, f64));
}
// Some targets have special rules for arrays.
#[cfg(not(any(target_arch = "mips64", target_arch = "sparc64")))]
mod arrays {
    use super::*;
    test_transparent!(empty_array, [u32; 0]);
    test_transparent!(empty_1zst_array, [u8; 0]);
    test_transparent!(small_array, [i32; 2]); // chosen to fit into 64bit
    test_transparent!(large_array, [i32; 16]);
}

// Some tests with unsized types (not all wrappers are compatible with that).
macro_rules! test_transparent_unsized {
    ($name:ident, $t:ty) => {
        mod $name {
            use super::*;
            assert_abi_compatible!(wrap1, $t, Wrapper1<$t>);
            assert_abi_compatible!(wrap1_reprc, ReprC1<$t>, ReprC1<Wrapper1<$t>>);
            assert_abi_compatible!(wrap2, $t, Wrapper2<$t>);
            assert_abi_compatible!(wrap2_reprc, ReprC1<$t>, ReprC1<Wrapper2<$t>>);
        }
    };
}

#[cfg(not(any(target_arch = "mips64", target_arch = "sparc64")))]
mod unsized_ {
    use super::*;
    test_transparent_unsized!(str_, str);
    test_transparent_unsized!(slice, [u8]);
    test_transparent_unsized!(dyn_trait, dyn std::any::Any);
}

// RFC 3391 <https://rust-lang.github.io/rfcs/3391-result_ffi_guarantees.html>.
macro_rules! test_nonnull {
    ($name:ident, $t:ty) => {
        mod $name {
            use super::*;
            test_abi_compatible!(option, Option<$t>, $t);
            test_abi_compatible!(result_err_unit, Result<$t, ()>, $t);
            test_abi_compatible!(result_ok_unit, Result<(), $t>, $t);
            test_abi_compatible!(result_err_zst, Result<$t, Zst>, $t);
            test_abi_compatible!(result_ok_zst, Result<Zst, $t>, $t);
            test_abi_compatible!(result_err_arr, Result<$t, [i8; 0]>, $t);
            test_abi_compatible!(result_ok_arr, Result<[i8; 0], $t>, $t);
        }
    }
}

test_nonnull!(ref_, &i32);
test_nonnull!(mut_, &mut i32);
test_nonnull!(ref_unsized, &[i32]);
test_nonnull!(mut_unsized, &mut [i32]);
test_nonnull!(fn_, fn());
test_nonnull!(nonnull, NonNull<i32>);
test_nonnull!(nonnull_unsized, NonNull<dyn std::fmt::Debug>);
test_nonnull!(non_zero, NonZeroI32);

fn main() {}
