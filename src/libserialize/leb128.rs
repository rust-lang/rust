// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(target_pointer_width = "32")]
const USIZE_LEB128_SIZE: usize = 5;
#[cfg(target_pointer_width = "64")]
const USIZE_LEB128_SIZE: usize = 10;

macro_rules! leb128_size {
    (u16) => (3);
    (u32) => (5);
    (u64) => (10);
    (u128) => (19);
    (usize) => (USIZE_LEB128_SIZE);
}

macro_rules! impl_write_unsigned_leb128 {
    ($fn_name:ident, $int_ty:ident) => (
        #[inline]
        pub fn $fn_name(out: &mut Vec<u8>, mut value: $int_ty) {
            out.extend((0..leb128_size!($int_ty)).scan(value, |state, _| {
                let mut byte = (*state & 0x7F) as u8;
                *state >>= 7;
                if *state != 0 {
                    byte |= 0x80;
                    Some(byte)
                } else {
                    value = byte as $int_ty;
                    None
                }
            }));
            out.push(value as u8);
        }
    )
}

impl_write_unsigned_leb128!(write_u16_leb128, u16);
impl_write_unsigned_leb128!(write_u32_leb128, u32);
impl_write_unsigned_leb128!(write_u64_leb128, u64);
impl_write_unsigned_leb128!(write_u128_leb128, u128);
impl_write_unsigned_leb128!(write_usize_leb128, usize);

macro_rules! impl_read_unsigned_leb128 {
    ($fn_name:ident, $int_ty:ident) => (
        #[inline]
        pub fn $fn_name(slice: &[u8]) -> ($int_ty, usize) {
            let mut result: $int_ty = 0;
            let mut shift = 0;
            let mut position = 0;

            for _ in 0 .. leb128_size!($int_ty) {
                let byte = unsafe {
                    *slice.get_unchecked(position)
                };
                position += 1;
                result |= ((byte & 0x7F) as $int_ty) << shift;
                if (byte & 0x80) == 0 {
                    break;
                }
                shift += 7;
            }

            // Do a single bounds check at the end instead of for every byte.
            assert!(position <= slice.len());

            (result, position)
        }
    )
}

impl_read_unsigned_leb128!(read_u16_leb128, u16);
impl_read_unsigned_leb128!(read_u32_leb128, u32);
impl_read_unsigned_leb128!(read_u64_leb128, u64);
impl_read_unsigned_leb128!(read_u128_leb128, u128);
impl_read_unsigned_leb128!(read_usize_leb128, usize);

#[inline]
/// encodes an integer using signed leb128 encoding and stores
/// the result using a callback function.
///
/// The callback `write` is called once for each position
/// that is to be written to with the byte to be encoded
/// at that position.
pub fn write_signed_leb128(out: &mut Vec<u8>, mut value: i128) {
    out.extend((0..).scan(value, |state, _| {
        let mut byte = (*state as u8) & 0x7f;
        *state >>= 7;
        let more = !((((*state ==  0) && ((byte & 0x40) == 0)) ||
                      ((*state == -1) && ((byte & 0x40) != 0))));

        if more {
            byte |= 0x80; // Mark this byte to show that more bytes will follow.
            Some(byte)
        } else {
            value = byte as i128;
            None
        }
    }));
    out.push(value as u8);
}

#[inline]
pub fn read_signed_leb128(data: &[u8], start_position: usize) -> (i128, usize) {
    let mut result = 0;
    let mut shift = 0;
    let mut position = start_position;
    let mut byte;

    loop {
        byte = data[position];
        position += 1;
        result |= ((byte & 0x7F) as i128) << shift;
        shift += 7;

        if (byte & 0x80) == 0 {
            break;
        }
    }

    if (shift < 64) && ((byte & 0x40) != 0) {
        // sign extend
        result |= -(1 << shift);
    }

    (result, position - start_position)
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use super::*;

    macro_rules! impl_test_unsigned_leb128 {
        ($test_name:ident, $write_fn_name:ident, $read_fn_name:ident, $int_ty:ident) => (
            #[test]
            fn $test_name() {
                let mut stream = Vec::new();

                for x in 0..62 {
                    $write_fn_name(&mut stream, (3u64 << x) as $int_ty);
                }

                let mut position = 0;
                for x in 0..62 {
                    let expected = (3u64 << x) as $int_ty;
                    let (actual, bytes_read) = $read_fn_name(&stream[position ..]);
                    assert_eq!(expected, actual);
                    position += bytes_read;
                }
                assert_eq!(stream.len(), position);
            }
        )
    }

    impl_test_unsigned_leb128!(test_u16_leb128, write_u16_leb128, read_u16_leb128, u16);
    impl_test_unsigned_leb128!(test_u32_leb128, write_u32_leb128, read_u32_leb128, u32);
    impl_test_unsigned_leb128!(test_u64_leb128, write_u64_leb128, read_u64_leb128, u64);
    impl_test_unsigned_leb128!(test_u128_leb128, write_u128_leb128, read_u128_leb128, u128);
    impl_test_unsigned_leb128!(test_usize_leb128, write_usize_leb128, read_usize_leb128, usize);

    #[test]
    fn test_signed_leb128() {
        let values: Vec<_> = (-500..500).map(|i| i * 0x12345789ABCDEF).collect();
        let mut stream = Vec::new();
        for &x in &values {
            write_signed_leb128(&mut stream, x);
        }
        let mut pos = 0;
        for &x in &values {
            let (value, bytes_read) = read_signed_leb128(&mut stream, pos);
            pos += bytes_read;
            assert_eq!(x, value);
        }
        assert_eq!(pos, stream.len());
    }

    macro_rules! impl_bench_leb128_write {
        ($bench_name:ident, $write_fn_name:ident, $int_ty:ident) => (
            #[bench]
            fn $bench_name(b: &mut Bencher) {
                b.iter(|| {
                    let mut stream = Vec::new();
                    for i in 0..1000 { $write_fn_name(&mut stream, i as $int_ty) }
                })
            }
        )
    }

    impl_bench_leb128_write!(bench_write_u16, write_u16_leb128, u16);
    impl_bench_leb128_write!(bench_write_u32, write_u32_leb128, u32);
    impl_bench_leb128_write!(bench_write_u64, write_u64_leb128, u64);
    impl_bench_leb128_write!(bench_write_u128, write_u128_leb128, u128);
    impl_bench_leb128_write!(bench_write_usize, write_usize_leb128, usize);
    impl_bench_leb128_write!(bench_write_i128, write_signed_leb128, i128);
}
