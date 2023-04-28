// run-pass

use std::mem;
use std::num::*;

macro_rules! define_with_type {
    ($namespace: ident, $ty: ty, $abi_ty: ty) => {
        mod $namespace {
            #[allow(unused)]
            use super::*;

            #[inline(never)]
            pub extern "C" fn option(is_some: bool, value: $ty) -> Option<$ty> {
                if is_some {
                    Some(value)
                } else {
                    None
                }
            }

            #[inline(never)]
            pub extern "C" fn result_pos(is_ok: bool, value: $ty) -> Result<$ty, ()> {
                if is_ok {
                    Ok(value)
                } else {
                    Err(())
                }
            }

            #[inline(never)]
            pub extern "C" fn result_neg(is_ok: bool, value: $ty) -> Result<(), $ty> {
                if is_ok {
                    Ok(())
                } else {
                    Err(value)
                }
            }

            #[inline(never)]
            pub extern "C" fn option_param(value: Option<$ty>) -> $abi_ty {
                unsafe { mem::transmute(value) }
            }

            #[inline(never)]
            pub extern "C" fn result_param_pos(value: Result<$ty, ()>) -> $abi_ty {
                unsafe { mem::transmute(value) }
            }

            #[inline(never)]
            pub extern "C" fn result_param_neg(value: Result<(), $ty>) -> $abi_ty {
                unsafe { mem::transmute(value) }
            }
        }
    };
}

define_with_type!(nonzero_i8, NonZeroI8, i8);
define_with_type!(nonzero_i16, NonZeroI16, i16);
define_with_type!(nonzero_i32, NonZeroI32, i32);
define_with_type!(nonzero_i64, NonZeroI64, i64);
define_with_type!(nonzero_u8, NonZeroU8, u8);
define_with_type!(nonzero_u16, NonZeroU16, u16);
define_with_type!(nonzero_u32, NonZeroU32, u32);
define_with_type!(nonzero_u64, NonZeroU64, u64);
define_with_type!(nonzero_usize, NonZeroUsize, usize);
define_with_type!(nonzero_ref, &'static i32, *const i32);

pub fn main() {
    macro_rules! test_with_type {
        (
            $namespace: ident,
            $ty: ty,
            $abi_ty: ty,
            $in_value: expr,
            $out_value: expr,
            $null_value:expr
        ) => {
            let in_value: $ty = $in_value;
            let out_value: $abi_ty = $out_value;
            let null_value: $abi_ty = $null_value;
            unsafe {
                let f: extern "C" fn(bool, $ty) -> $abi_ty =
                    mem::transmute($namespace::option as extern "C" fn(bool, $ty) -> Option<$ty>);
                assert_eq!(f(true, in_value), out_value);
                assert_eq!(f(false, in_value), null_value);
            }

            unsafe {
                let f: extern "C" fn(bool, $ty) -> $abi_ty = mem::transmute(
                    $namespace::result_pos as extern "C" fn(bool, $ty) -> Result<$ty, ()>,
                );
                assert_eq!(f(true, in_value), out_value);
                assert_eq!(f(false, in_value), null_value);
            }

            unsafe {
                let f: extern "C" fn(bool, $ty) -> $abi_ty = mem::transmute(
                    $namespace::result_neg as extern "C" fn(bool, $ty) -> Result<(), $ty>,
                );
                assert_eq!(f(false, in_value), out_value);
                assert_eq!(f(true, in_value), null_value);
            }

            assert_eq!($namespace::option_param(Some(in_value)), out_value);
            assert_eq!($namespace::option_param(None), null_value);

            assert_eq!($namespace::result_param_pos(Ok(in_value)), out_value);
            assert_eq!($namespace::result_param_pos(Err(())), null_value);

            assert_eq!($namespace::result_param_neg(Err(in_value)), out_value);
            assert_eq!($namespace::result_param_neg(Ok(())), null_value);
        };
    }
    test_with_type!(nonzero_i8, NonZeroI8, i8, NonZeroI8::new(123).unwrap(), 123, 0);
    test_with_type!(nonzero_i16, NonZeroI16, i16, NonZeroI16::new(1234).unwrap(), 1234, 0);
    test_with_type!(nonzero_i32, NonZeroI32, i32, NonZeroI32::new(1234).unwrap(), 1234, 0);
    test_with_type!(nonzero_i64, NonZeroI64, i64, NonZeroI64::new(1234).unwrap(), 1234, 0);
    test_with_type!(nonzero_u8, NonZeroU8, u8, NonZeroU8::new(123).unwrap(), 123, 0);
    test_with_type!(nonzero_u16, NonZeroU16, u16, NonZeroU16::new(1234).unwrap(), 1234, 0);
    test_with_type!(nonzero_u32, NonZeroU32, u32, NonZeroU32::new(1234).unwrap(), 1234, 0);
    test_with_type!(nonzero_u64, NonZeroU64, u64, NonZeroU64::new(1234).unwrap(), 1234, 0);
    test_with_type!(nonzero_usize, NonZeroUsize, usize, NonZeroUsize::new(1234).unwrap(), 1234, 0);
    static FOO: i32 = 0xDEADBEE;
    test_with_type!(nonzero_ref, &'static i32, *const i32, &FOO, &FOO, std::ptr::null());
}
