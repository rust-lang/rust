// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utility mixins that apply to all Readers and Writers

// XXX: Not sure how this should be structured
// XXX: Iteration should probably be considered separately

pub fn u64_to_le_bytes<T>(n: u64, size: uint,
                          f: |v: &[u8]| -> T) -> T {
    assert!(size <= 8u);
    match size {
      1u => f(&[n as u8]),
      2u => f(&[n as u8,
              (n >> 8) as u8]),
      4u => f(&[n as u8,
              (n >> 8) as u8,
              (n >> 16) as u8,
              (n >> 24) as u8]),
      8u => f(&[n as u8,
              (n >> 8) as u8,
              (n >> 16) as u8,
              (n >> 24) as u8,
              (n >> 32) as u8,
              (n >> 40) as u8,
              (n >> 48) as u8,
              (n >> 56) as u8]),
      _ => {

        let mut bytes: ~[u8] = ~[];
        let mut i = size;
        let mut n = n;
        while i > 0u {
            bytes.push((n & 255_u64) as u8);
            n >>= 8_u64;
            i -= 1u;
        }
        f(bytes)
      }
    }
}

pub fn u64_to_be_bytes<T>(n: u64, size: uint, f: |v: &[u8]| -> T) -> T {
    assert!(size <= 8u);
    match size {
      1u => f(&[n as u8]),
      2u => f(&[(n >> 8) as u8,
              n as u8]),
      4u => f(&[(n >> 24) as u8,
              (n >> 16) as u8,
              (n >> 8) as u8,
              n as u8]),
      8u => f(&[(n >> 56) as u8,
              (n >> 48) as u8,
              (n >> 40) as u8,
              (n >> 32) as u8,
              (n >> 24) as u8,
              (n >> 16) as u8,
              (n >> 8) as u8,
              n as u8]),
      _ => {
        let mut bytes: ~[u8] = ~[];
        let mut i = size;
        while i > 0u {
            let shift = ((i - 1u) * 8u) as u64;
            bytes.push((n >> shift) as u8);
            i -= 1u;
        }
        f(bytes)
      }
    }
}

pub fn u64_from_be_bytes(data: &[u8],
                         start: uint,
                         size: uint)
                      -> u64 {
    let mut sz = size;
    assert!((sz <= 8u));
    let mut val = 0_u64;
    let mut pos = start;
    while sz > 0u {
        sz -= 1u;
        val += (data[pos] as u64) << ((sz * 8u) as u64);
        pos += 1u;
    }
    return val;
}

#[cfg(test)]
mod test {
    use result::{Ok, Err};
    use option::{None, Some};
    use io::mem::{MemReader, MemWriter};
    use io::{Reader, standard_error, EndOfFile, IoUnavailable, IoResult};
    use vec::ImmutableVector;

    struct InitialZeroByteReader {
        count: int,
    }

    impl Reader for InitialZeroByteReader {
        fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
            if self.count == 0 {
                self.count = 1;
                Ok(0)
            } else {
                buf[0] = 10;
                Ok(1)
            }
        }
        fn eof(&mut self) -> bool {
            false
        }
    }

    struct EofReader;

    impl Reader for EofReader {
        fn read(&mut self, _: &mut [u8]) -> IoResult<uint> {
            Err(standard_error(EndOfFile))
        }
        fn eof(&mut self) -> bool {
            false
        }
    }

    struct ErroringReader;

    impl Reader for ErroringReader {
        fn read(&mut self, _: &mut [u8]) -> IoResult<uint> {
            Err(standard_error(IoUnavailable))
        }
        fn eof(&mut self) -> bool {
            false
        }
    }

    struct PartialReader {
        count: int,
    }

    impl Reader for PartialReader {
        fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
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
        fn eof(&mut self) -> bool {
            false
        }
    }

    struct ErroringLaterReader {
        count: int,
    }

    impl Reader for ErroringLaterReader {
        fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
            if self.count == 0 {
                self.count = 1;
                buf[0] = 10;
                Ok(1)
            } else {
                Err(standard_error(IoUnavailable))
            }
        }
        fn eof(&mut self) -> bool {
            false
        }
    }

    struct ThreeChunkReader {
        count: int,
    }

    impl Reader for ThreeChunkReader {
        fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
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
                Err(standard_error(EndOfFile))
            }
        }
        fn eof(&mut self) -> bool {
            false
        }
    }

    #[test]
    fn read_byte() {
        let mut reader = MemReader::new(~[10]);
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
    fn read_bytes() {
        let mut reader = MemReader::new(~[10, 11, 12, 13]);
        let bytes = reader.read_bytes(4);
        assert!(bytes == Ok(~[10, 11, 12, 13]));
    }

    #[test]
    fn read_bytes_partial() {
        let mut reader = PartialReader {
            count: 0,
        };
        let bytes = reader.read_bytes(4);
        assert!(bytes == Ok(~[10, 11, 12, 13]));
    }

    #[test]
    fn read_bytes_eof() {
        let mut reader = MemReader::new(~[10, 11]);
        match reader.read_bytes(4) {
            Ok(*) => fail!(),
            Err((b, _)) => assert_eq!(b, ~[10, 11]),
        }
    }

    #[test]
    fn push_bytes() {
        let mut reader = MemReader::new(~[10, 11, 12, 13]);
        let mut buf = ~[8, 9];
        reader.push_bytes(&mut buf, 4);
        assert!(buf == ~[8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn push_bytes_partial() {
        let mut reader = PartialReader {
            count: 0,
        };
        let mut buf = ~[8, 9];
        reader.push_bytes(&mut buf, 4);
        assert!(buf == ~[8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn push_bytes_eof() {
        let mut reader = MemReader::new(~[10, 11]);
        let mut buf = ~[8, 9];
        reader.push_bytes(&mut buf, 4);
        assert!(buf == ~[8, 9, 10, 11]);
    }

    #[test]
    fn push_bytes_error() {
        let mut reader = ErroringLaterReader {
            count: 0,
        };
        let mut buf = ~[8, 9];
        reader.push_bytes(&mut buf, 4);
        assert!(buf == ~[8, 9, 10]);
    }

    #[test]
    #[should_fail]
    #[ignore] // FIXME(#7049): buf is still borrowed
    fn push_bytes_fail_reset_len() {
        // push_bytes unsafely sets the vector length. This is testing that
        // upon failure the length is reset correctly.
        let mut reader = ErroringLaterReader {
            count: 0,
        };
        let buf = @mut ~[8, 9];
        (|| {
            reader.push_bytes(&mut *buf, 4);
        }).finally(|| {
            // NB: Using rtassert here to trigger abort on failure since this is a should_fail test
            // FIXME: #7049 This fails because buf is still borrowed
            //rtassert!(*buf == ~[8, 9, 10]);
        })
    }

    #[test]
    fn read_to_end() {
        let mut reader = ThreeChunkReader {
            count: 0,
        };
        let buf = reader.read_to_end();
        assert!(buf == Ok(~[10, 11, 12, 13]));
    }

    #[test]
    #[should_fail]
    fn read_to_end_error() {
        let mut reader = ThreeChunkReader {
            count: 0,
        };
        let buf = reader.read_to_end();
        assert!(buf == Ok(~[10, 11]));
    }

    #[test]
    fn test_read_write_le_mem() {
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, ::u64::max_value];

        let mut writer = MemWriter::new();
        for i in uints.iter() {
            writer.write_le_u64(*i);
        }

        let mut reader = MemReader::new(writer.inner());
        for i in uints.iter() {
            assert!(reader.read_le_u64() == Ok(*i));
        }
    }


    #[test]
    fn test_read_write_be() {
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, ::u64::max_value];

        let mut writer = MemWriter::new();
        for i in uints.iter() {
            writer.write_be_u64(*i);
        }

        let mut reader = MemReader::new(writer.inner());
        for i in uints.iter() {
            assert!(reader.read_be_u64() == Ok(*i));
        }
    }

    #[test]
    fn test_read_be_int_n() {
        let ints = [::i32::min_value, -123456, -42, -5, 0, 1, ::i32::max_value];

        let mut writer = MemWriter::new();
        for i in ints.iter() {
            writer.write_be_i32(*i);
        }

        let mut reader = MemReader::new(writer.inner());
        for i in ints.iter() {
            // this tests that the sign extension is working
            // (comparing the values as i32 would not test this)
            assert!(reader.read_be_int_n(4) == Ok(*i as i64));
        }
    }

    #[test]
    fn test_read_f32() {
        //big-endian floating-point 8.1250
        let buf = ~[0x41, 0x02, 0x00, 0x00];

        let mut writer = MemWriter::new();
        writer.write(buf);

        let mut reader = MemReader::new(writer.inner());
        let f = reader.read_be_f32();
        assert!(f == Ok(8.1250));
    }

    #[test]
    fn test_read_write_f32() {
        let f:f32 = 8.1250;

        let mut writer = MemWriter::new();
        writer.write_be_f32(f);
        writer.write_le_f32(f);

        let mut reader = MemReader::new(writer.inner());
        assert!(reader.read_be_f32() == Ok(8.1250));
        assert!(reader.read_le_f32() == Ok(8.1250));
    }

}
