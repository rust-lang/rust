#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_uninit_array)]

use rustc_serialize::leb128::*;
use std::mem::MaybeUninit;

macro_rules! impl_test_unsigned_leb128 {
    ($test_name:ident, $write_fn_name:ident, $read_fn_name:ident, $int_ty:ident) => {
        #[test]
        fn $test_name() {
            // Test 256 evenly spaced values of integer range,
            // integer max value, and some "random" numbers.
            let mut values = Vec::new();

            let increment = (1 as $int_ty) << ($int_ty::BITS - 8);
            values.extend((0..256).map(|i| $int_ty::MIN + i * increment));

            values.push($int_ty::MAX);

            values.extend(
                (-500..500).map(|i| (i as $int_ty).wrapping_mul(0x12345789ABCDEFu64 as $int_ty)),
            );

            let mut stream = Vec::new();

            for &x in &values {
                let mut buf = MaybeUninit::uninit_array();
                stream.extend($write_fn_name(&mut buf, x));
            }

            let mut position = 0;
            for &expected in &values {
                let actual = $read_fn_name(&stream, &mut position);
                assert_eq!(expected, actual);
            }
            assert_eq!(stream.len(), position);
        }
    };
}

impl_test_unsigned_leb128!(test_u16_leb128, write_u16_leb128, read_u16_leb128, u16);
impl_test_unsigned_leb128!(test_u32_leb128, write_u32_leb128, read_u32_leb128, u32);
impl_test_unsigned_leb128!(test_u64_leb128, write_u64_leb128, read_u64_leb128, u64);
impl_test_unsigned_leb128!(test_u128_leb128, write_u128_leb128, read_u128_leb128, u128);
impl_test_unsigned_leb128!(test_usize_leb128, write_usize_leb128, read_usize_leb128, usize);

macro_rules! impl_test_signed_leb128 {
    ($test_name:ident, $write_fn_name:ident, $read_fn_name:ident, $int_ty:ident) => {
        #[test]
        fn $test_name() {
            // Test 256 evenly spaced values of integer range,
            // integer max value, and some "random" numbers.
            let mut values = Vec::new();

            let mut value = $int_ty::MIN;
            let increment = (1 as $int_ty) << ($int_ty::BITS - 8);

            for _ in 0..256 {
                values.push(value);
                // The addition in the last loop iteration overflows.
                value = value.wrapping_add(increment);
            }

            values.push($int_ty::MAX);

            values.extend(
                (-500..500).map(|i| (i as $int_ty).wrapping_mul(0x12345789ABCDEFi64 as $int_ty)),
            );

            let mut stream = Vec::new();

            for &x in &values {
                let mut buf = MaybeUninit::uninit_array();
                stream.extend($write_fn_name(&mut buf, x));
            }

            let mut position = 0;
            for &expected in &values {
                let actual = $read_fn_name(&stream, &mut position);
                assert_eq!(expected, actual);
            }
            assert_eq!(stream.len(), position);
        }
    };
}

impl_test_signed_leb128!(test_i16_leb128, write_i16_leb128, read_i16_leb128, i16);
impl_test_signed_leb128!(test_i32_leb128, write_i32_leb128, read_i32_leb128, i32);
impl_test_signed_leb128!(test_i64_leb128, write_i64_leb128, read_i64_leb128, i64);
impl_test_signed_leb128!(test_i128_leb128, write_i128_leb128, read_i128_leb128, i128);
impl_test_signed_leb128!(test_isize_leb128, write_isize_leb128, read_isize_leb128, isize);
