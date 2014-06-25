// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utility mixins that apply to all Readers and Writers

#![allow(missing_doc)]

// FIXME: Not sure how this should be structured
// FIXME: Iteration should probably be considered separately

use collections::Collection;
use iter::Iterator;
use option::{Option, Some, None};
use result::{Ok, Err};
use io;
use io::{IoError, IoResult, Reader};
use slice::{ImmutableVector, Vector};
use ptr::RawPtr;

/// An iterator that reads a single byte on each iteration,
/// until `.read_byte()` returns `EndOfFile`.
///
/// # Notes about the Iteration Protocol
///
/// The `Bytes` may yield `None` and thus terminate
/// an iteration, but continue to yield elements if iteration
/// is attempted again.
///
/// # Error
///
/// Any error other than `EndOfFile` that is produced by the underlying Reader
/// is returned by the iterator and should be handled by the caller.
pub struct Bytes<'r, T> {
    reader: &'r mut T,
}

impl<'r, R: Reader> Bytes<'r, R> {
    /// Constructs a new byte iterator from the given Reader instance.
    pub fn new(r: &'r mut R) -> Bytes<'r, R> {
        Bytes {
            reader: r,
        }
    }
}

impl<'r, R: Reader> Iterator<IoResult<u8>> for Bytes<'r, R> {
    #[inline]
    fn next(&mut self) -> Option<IoResult<u8>> {
        match self.reader.read_byte() {
            Ok(x) => Some(Ok(x)),
            Err(IoError { kind: io::EndOfFile, .. }) => None,
            Err(e) => Some(Err(e))
        }
    }
}

/// Converts an 8-bit to 64-bit unsigned value to a little-endian byte
/// representation of the given size. If the size is not big enough to
/// represent the value, then the high-order bytes are truncated.
///
/// Arguments:
///
/// * `n`: The value to convert.
/// * `size`: The size of the value, in bytes. This must be 8 or less, or task
///           failure occurs. If this is less than 8, then a value of that
///           many bytes is produced. For example, if `size` is 4, then a
///           32-bit byte representation is produced.
/// * `f`: A callback that receives the value.
///
/// This function returns the value returned by the callback, for convenience.
pub fn u64_to_le_bytes<T>(n: u64, size: uint, f: |v: &[u8]| -> T) -> T {
    use mem::{to_le16, to_le32, to_le64};
    use mem::transmute;

    // LLVM fails to properly optimize this when using shifts instead of the to_le* intrinsics
    assert!(size <= 8u);
    match size {
      1u => f(&[n as u8]),
      2u => f(unsafe { transmute::<_, [u8, ..2]>(to_le16(n as u16)) }),
      4u => f(unsafe { transmute::<_, [u8, ..4]>(to_le32(n as u32)) }),
      8u => f(unsafe { transmute::<_, [u8, ..8]>(to_le64(n)) }),
      _ => {

        let mut bytes = vec!();
        let mut i = size;
        let mut n = n;
        while i > 0u {
            bytes.push((n & 255_u64) as u8);
            n >>= 8;
            i -= 1u;
        }
        f(bytes.as_slice())
      }
    }
}

/// Converts an 8-bit to 64-bit unsigned value to a big-endian byte
/// representation of the given size. If the size is not big enough to
/// represent the value, then the high-order bytes are truncated.
///
/// Arguments:
///
/// * `n`: The value to convert.
/// * `size`: The size of the value, in bytes. This must be 8 or less, or task
///           failure occurs. If this is less than 8, then a value of that
///           many bytes is produced. For example, if `size` is 4, then a
///           32-bit byte representation is produced.
/// * `f`: A callback that receives the value.
///
/// This function returns the value returned by the callback, for convenience.
pub fn u64_to_be_bytes<T>(n: u64, size: uint, f: |v: &[u8]| -> T) -> T {
    use mem::{to_be16, to_be32, to_be64};
    use mem::transmute;

    // LLVM fails to properly optimize this when using shifts instead of the to_be* intrinsics
    assert!(size <= 8u);
    match size {
      1u => f(&[n as u8]),
      2u => f(unsafe { transmute::<_, [u8, ..2]>(to_be16(n as u16)) }),
      4u => f(unsafe { transmute::<_, [u8, ..4]>(to_be32(n as u32)) }),
      8u => f(unsafe { transmute::<_, [u8, ..8]>(to_be64(n)) }),
      _ => {
        let mut bytes = vec!();
        let mut i = size;
        while i > 0u {
            let shift = (i - 1u) * 8u;
            bytes.push((n >> shift) as u8);
            i -= 1u;
        }
        f(bytes.as_slice())
      }
    }
}

/// Extracts an 8-bit to 64-bit unsigned big-endian value from the given byte
/// buffer and returns it as a 64-bit value.
///
/// Arguments:
///
/// * `data`: The buffer in which to extract the value.
/// * `start`: The offset at which to extract the value.
/// * `size`: The size of the value in bytes to extract. This must be 8 or
///           less, or task failure occurs. If this is less than 8, then only
///           that many bytes are parsed. For example, if `size` is 4, then a
///           32-bit value is parsed.
pub fn u64_from_be_bytes(data: &[u8], start: uint, size: uint) -> u64 {
    use ptr::{copy_nonoverlapping_memory};
    use mem::from_be64;
    use slice::MutableVector;

    assert!(size <= 8u);

    if data.len() - start < size {
        fail!("index out of bounds");
    }

    let mut buf = [0u8, ..8];
    unsafe {
        let ptr = data.as_ptr().offset(start as int);
        let out = buf.as_mut_ptr();
        copy_nonoverlapping_memory(out.offset((8 - size) as int), ptr, size);
        from_be64(*(out as *u64))
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use io;
    use io::{MemReader, MemWriter};

    struct InitialZeroByteReader {
        count: int,
    }

    impl Reader for InitialZeroByteReader {
        fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
            if self.count == 0 {
                self.count = 1;
                Ok(0)
            } else {
                buf[0] = 10;
                Ok(1)
            }
        }
    }

    struct EofReader;

    impl Reader for EofReader {
        fn read(&mut self, _: &mut [u8]) -> io::IoResult<uint> {
            Err(io::standard_error(io::EndOfFile))
        }
    }

    struct ErroringReader;

    impl Reader for ErroringReader {
        fn read(&mut self, _: &mut [u8]) -> io::IoResult<uint> {
            Err(io::standard_error(io::InvalidInput))
        }
    }

    struct PartialReader {
        count: int,
    }

    impl Reader for PartialReader {
        fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
            if self.count == 0 {
                self.count = 1;
                buf[0] = 10;
                buf[1] = 11;
                Ok(2)
            } else {
                buf[0] = 12;
                buf[1] = 13;
                Ok(2)
            }
        }
    }

    struct ErroringLaterReader {
        count: int,
    }

    impl Reader for ErroringLaterReader {
        fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
            if self.count == 0 {
                self.count = 1;
                buf[0] = 10;
                Ok(1)
            } else {
                Err(io::standard_error(io::InvalidInput))
            }
        }
    }

    struct ThreeChunkReader {
        count: int,
    }

    impl Reader for ThreeChunkReader {
        fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
            if self.count == 0 {
                self.count = 1;
                buf[0] = 10;
                buf[1] = 11;
                Ok(2)
            } else if self.count == 1 {
                self.count = 2;
                buf[0] = 12;
                buf[1] = 13;
                Ok(2)
            } else {
                Err(io::standard_error(io::EndOfFile))
            }
        }
    }

    #[test]
    fn read_byte() {
        let mut reader = MemReader::new(vec!(10));
        let byte = reader.read_byte();
        assert!(byte == Ok(10));
    }

    #[test]
    fn read_byte_0_bytes() {
        let mut reader = InitialZeroByteReader {
            count: 0,
        };
        let byte = reader.read_byte();
        assert!(byte == Ok(10));
    }

    #[test]
    fn read_byte_eof() {
        let mut reader = EofReader;
        let byte = reader.read_byte();
        assert!(byte.is_err());
    }

    #[test]
    fn read_byte_error() {
        let mut reader = ErroringReader;
        let byte = reader.read_byte();
        assert!(byte.is_err());
    }

    #[test]
    fn bytes_0_bytes() {
        let mut reader = InitialZeroByteReader {
            count: 0,
        };
        let byte = reader.bytes().next();
        assert!(byte == Some(Ok(10)));
    }

    #[test]
    fn bytes_eof() {
        let mut reader = EofReader;
        let byte = reader.bytes().next();
        assert!(byte.is_none());
    }

    #[test]
    fn bytes_error() {
        let mut reader = ErroringReader;
        let mut it = reader.bytes();
        let byte = it.next();
        assert!(byte.unwrap().is_err());
    }

    #[test]
    fn read_bytes() {
        let mut reader = MemReader::new(vec!(10, 11, 12, 13));
        let bytes = reader.read_exact(4).unwrap();
        assert!(bytes == vec!(10, 11, 12, 13));
    }

    #[test]
    fn read_bytes_partial() {
        let mut reader = PartialReader {
            count: 0,
        };
        let bytes = reader.read_exact(4).unwrap();
        assert!(bytes == vec!(10, 11, 12, 13));
    }

    #[test]
    fn read_bytes_eof() {
        let mut reader = MemReader::new(vec!(10, 11));
        assert!(reader.read_exact(4).is_err());
    }

    #[test]
    fn push_at_least() {
        let mut reader = MemReader::new(vec![10, 11, 12, 13]);
        let mut buf = vec![8, 9];
        assert!(reader.push_at_least(4, 4, &mut buf).is_ok());
        assert!(buf == vec![8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn push_at_least_partial() {
        let mut reader = PartialReader {
            count: 0,
        };
        let mut buf = vec![8, 9];
        assert!(reader.push_at_least(4, 4, &mut buf).is_ok());
        assert!(buf == vec![8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn push_at_least_eof() {
        let mut reader = MemReader::new(vec![10, 11]);
        let mut buf = vec![8, 9];
        assert!(reader.push_at_least(4, 4, &mut buf).is_err());
        assert!(buf == vec![8, 9, 10, 11]);
    }

    #[test]
    fn push_at_least_error() {
        let mut reader = ErroringLaterReader {
            count: 0,
        };
        let mut buf = vec![8, 9];
        assert!(reader.push_at_least(4, 4, &mut buf).is_err());
        assert!(buf == vec![8, 9, 10]);
    }

    #[test]
    fn read_to_end() {
        let mut reader = ThreeChunkReader {
            count: 0,
        };
        let buf = reader.read_to_end().unwrap();
        assert!(buf == vec!(10, 11, 12, 13));
    }

    #[test]
    #[should_fail]
    fn read_to_end_error() {
        let mut reader = ThreeChunkReader {
            count: 0,
        };
        let buf = reader.read_to_end().unwrap();
        assert!(buf == vec!(10, 11));
    }

    #[test]
    fn test_read_write_le_mem() {
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, ::u64::MAX];

        let mut writer = MemWriter::new();
        for i in uints.iter() {
            writer.write_le_u64(*i).unwrap();
        }

        let mut reader = MemReader::new(writer.unwrap());
        for i in uints.iter() {
            assert!(reader.read_le_u64().unwrap() == *i);
        }
    }


    #[test]
    fn test_read_write_be() {
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, ::u64::MAX];

        let mut writer = MemWriter::new();
        for i in uints.iter() {
            writer.write_be_u64(*i).unwrap();
        }

        let mut reader = MemReader::new(writer.unwrap());
        for i in uints.iter() {
            assert!(reader.read_be_u64().unwrap() == *i);
        }
    }

    #[test]
    fn test_read_be_int_n() {
        let ints = [::i32::MIN, -123456, -42, -5, 0, 1, ::i32::MAX];

        let mut writer = MemWriter::new();
        for i in ints.iter() {
            writer.write_be_i32(*i).unwrap();
        }

        let mut reader = MemReader::new(writer.unwrap());
        for i in ints.iter() {
            // this tests that the sign extension is working
            // (comparing the values as i32 would not test this)
            assert!(reader.read_be_int_n(4).unwrap() == *i as i64);
        }
    }

    #[test]
    fn test_read_f32() {
        //big-endian floating-point 8.1250
        let buf = vec![0x41, 0x02, 0x00, 0x00];

        let mut writer = MemWriter::new();
        writer.write(buf.as_slice()).unwrap();

        let mut reader = MemReader::new(writer.unwrap());
        let f = reader.read_be_f32().unwrap();
        assert!(f == 8.1250);
    }

    #[test]
    fn test_read_write_f32() {
        let f:f32 = 8.1250;

        let mut writer = MemWriter::new();
        writer.write_be_f32(f).unwrap();
        writer.write_le_f32(f).unwrap();

        let mut reader = MemReader::new(writer.unwrap());
        assert!(reader.read_be_f32().unwrap() == 8.1250);
        assert!(reader.read_le_f32().unwrap() == 8.1250);
    }

    #[test]
    fn test_u64_from_be_bytes() {
        use super::u64_from_be_bytes;

        let buf = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09];

        // Aligned access
        assert_eq!(u64_from_be_bytes(buf, 0, 0), 0);
        assert_eq!(u64_from_be_bytes(buf, 0, 1), 0x01);
        assert_eq!(u64_from_be_bytes(buf, 0, 2), 0x0102);
        assert_eq!(u64_from_be_bytes(buf, 0, 3), 0x010203);
        assert_eq!(u64_from_be_bytes(buf, 0, 4), 0x01020304);
        assert_eq!(u64_from_be_bytes(buf, 0, 5), 0x0102030405);
        assert_eq!(u64_from_be_bytes(buf, 0, 6), 0x010203040506);
        assert_eq!(u64_from_be_bytes(buf, 0, 7), 0x01020304050607);
        assert_eq!(u64_from_be_bytes(buf, 0, 8), 0x0102030405060708);

        // Unaligned access
        assert_eq!(u64_from_be_bytes(buf, 1, 0), 0);
        assert_eq!(u64_from_be_bytes(buf, 1, 1), 0x02);
        assert_eq!(u64_from_be_bytes(buf, 1, 2), 0x0203);
        assert_eq!(u64_from_be_bytes(buf, 1, 3), 0x020304);
        assert_eq!(u64_from_be_bytes(buf, 1, 4), 0x02030405);
        assert_eq!(u64_from_be_bytes(buf, 1, 5), 0x0203040506);
        assert_eq!(u64_from_be_bytes(buf, 1, 6), 0x020304050607);
        assert_eq!(u64_from_be_bytes(buf, 1, 7), 0x02030405060708);
        assert_eq!(u64_from_be_bytes(buf, 1, 8), 0x0203040506070809);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    use collections::Collection;
    use prelude::*;
    use self::test::Bencher;

    macro_rules! u64_from_be_bytes_bench_impl(
        ($size:expr, $stride:expr, $start_index:expr) =>
        ({
            use super::u64_from_be_bytes;

            let data = Vec::from_fn($stride*100+$start_index, |i| i as u8);
            let mut sum = 0u64;
            b.iter(|| {
                let mut i = $start_index;
                while i < data.len() {
                    sum += u64_from_be_bytes(data.as_slice(), i, $size);
                    i += $stride;
                }
            });
        })
    )

    #[bench]
    fn u64_from_be_bytes_4_aligned(b: &mut Bencher) {
        u64_from_be_bytes_bench_impl!(4, 4, 0);
    }

    #[bench]
    fn u64_from_be_bytes_4_unaligned(b: &mut Bencher) {
        u64_from_be_bytes_bench_impl!(4, 4, 1);
    }

    #[bench]
    fn u64_from_be_bytes_7_aligned(b: &mut Bencher) {
        u64_from_be_bytes_bench_impl!(7, 8, 0);
    }

    #[bench]
    fn u64_from_be_bytes_7_unaligned(b: &mut Bencher) {
        u64_from_be_bytes_bench_impl!(7, 8, 1);
    }

    #[bench]
    fn u64_from_be_bytes_8_aligned(b: &mut Bencher) {
        u64_from_be_bytes_bench_impl!(8, 8, 0);
    }

    #[bench]
    fn u64_from_be_bytes_8_unaligned(b: &mut Bencher) {
        u64_from_be_bytes_bench_impl!(8, 8, 1);
    }
}
