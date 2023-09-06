// check-pass
#![feature(rustc_attrs)]
#![allow(unused, improper_ctypes_definitions)]

use std::marker::PhantomData;

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
    }
}

#[derive(Copy, Clone)]
struct Zst;

// Check that various `transparent` wrappers result in equal ABIs.
#[repr(transparent)]
struct Wrapper1<T>(T);
#[repr(transparent)]
struct Wrapper2<T>((), Zst, T);
#[repr(transparent)]
struct Wrapper3<T>(T, [u8; 0], PhantomData<u64>);

#[repr(C)]
struct ReprCStruct<T>(T, f32, i32, T);
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

macro_rules! test_transparent {
    ($name:ident, $t:ty) => {
        mod $name {
            use super::*;
            assert_abi_compatible!(wrap1, $t, Wrapper1<$t>);
            assert_abi_compatible!(wrap2, $t, Wrapper2<$t>);
            assert_abi_compatible!(wrap3, $t, Wrapper3<$t>);
            // Also try adding some surrounding `repr(C)` types.
            assert_abi_compatible!(repr_c_struct_wrap1, ReprCStruct<$t>, ReprCStruct<Wrapper1<$t>>);
            assert_abi_compatible!(repr_c_enum_wrap1, ReprCEnum<$t>, ReprCEnum<Wrapper1<$t>>);
            assert_abi_compatible!(repr_c_union_wrap1, ReprCUnion<$t>, ReprCUnion<Wrapper1<$t>>);
            assert_abi_compatible!(repr_c_struct_wrap2, ReprCStruct<$t>, ReprCStruct<Wrapper2<$t>>);
            assert_abi_compatible!(repr_c_enum_wrap2, ReprCEnum<$t>, ReprCEnum<Wrapper2<$t>>);
            assert_abi_compatible!(repr_c_union_wrap2, ReprCUnion<$t>, ReprCUnion<Wrapper2<$t>>);
            assert_abi_compatible!(repr_c_struct_wrap3, ReprCStruct<$t>, ReprCStruct<Wrapper3<$t>>);
            assert_abi_compatible!(repr_c_enum_wrap3, ReprCEnum<$t>, ReprCEnum<Wrapper3<$t>>);
            assert_abi_compatible!(repr_c_union_wrap3, ReprCUnion<$t>, ReprCUnion<Wrapper3<$t>>);
        }
    }
}

test_transparent!(simple, i32);
test_transparent!(reference, &'static i32);
test_transparent!(zst, Zst);
test_transparent!(unit, ());
test_transparent!(pair, (i32, f32));
test_transparent!(triple, (i8, i16, f32)); // chosen to fit into 64bit
test_transparent!(tuple, (i32, f32, i64, f64));
test_transparent!(empty_array, [u32; 0]);
test_transparent!(empty_1zst_array, [u8; 0]);
test_transparent!(small_array, [i32; 2]); // chosen to fit into 64bit
test_transparent!(large_array, [i32; 16]);
test_transparent!(enum_, Option<i32>);
test_transparent!(enum_niched, Option<&'static i32>);

fn main() {}
