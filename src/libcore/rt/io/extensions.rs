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

use vec;
use rt::io::Reader;
use option::{Option, Some, None};
use unstable::finally::Finally;
use util;

pub trait ReaderUtil {

    /// Reads a single byte. Returns `None` on EOF.
    ///
    /// # Failure
    ///
    /// Raises the same conditions as the `read` method. Returns
    /// `None` if the condition is handled.
    fn read_byte(&mut self) -> Option<u8>;

    /// Reads `len` bytes and appends them to a vector.
    ///
    /// May push fewer than the requested number of bytes on error
    /// or EOF. Returns true on success, false on EOF or error.
    ///
    /// # Failure
    ///
    /// Raises the same conditions as `read`. Returns `false` if
    /// the condition is handled.
    fn push_bytes(&mut self, buf: &mut ~[u8], len: uint) -> bool;

    /// Reads `len` bytes and gives you back a new vector
    ///
    /// # Failure
    ///
    /// Raises the same conditions as the `read` method. May return
    /// less than the requested number of bytes on error or EOF.
    fn read_bytes(&mut self, len: uint) -> ~[u8];

    /// Reads all remaining bytes from the stream.
    ///
    /// # Failure
    ///
    /// Raises the same conditions as the `read` method.
    fn read_to_end(&mut self) -> ~[u8];

}

impl<T: Reader> ReaderUtil for T {
    fn read_byte(&mut self) -> Option<u8> {
        let mut buf = [0];
        match self.read(buf) {
            Some(0) => {
                debug!("read 0 bytes. trying again");
                self.read_byte()
            }
            Some(1) => Some(buf[0]),
            Some(_) => util::unreachable(),
            None => None
        }
    }

    fn push_bytes(&mut self, buf: &mut ~[u8], len: uint) -> bool {
        unsafe {
            let start_len = buf.len();
            let mut total_read = 0;
            let mut eof = false;

            vec::reserve_at_least(buf, start_len + len);
            vec::raw::set_len(buf, start_len + len);

            do (|| {
                while total_read < len {
                    let slice = vec::mut_slice(*buf, start_len + total_read, buf.len());
                    match self.read(slice) {
                        Some(nread) => {
                            total_read += nread;
                        }
                        None => {
                            eof = true;
                            break;
                        }
                    }
                }
            }).finally {
                vec::raw::set_len(buf, start_len + total_read);
            }

            return !eof;
        }
    }

    fn read_bytes(&mut self, len: uint) -> ~[u8] {
        let mut buf = vec::with_capacity(len);
        self.push_bytes(&mut buf, len);
        return buf;
    }

    fn read_to_end(&mut self) -> ~[u8] {
        fail!()
    }
}

pub trait ReaderByteConversions {
    /// Reads `n` little-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_uint_n(&mut self, nbytes: uint) -> u64;

    /// Reads `n` little-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_int_n(&mut self, nbytes: uint) -> i64;

    /// Reads `n` big-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_uint_n(&mut self, nbytes: uint) -> u64;

    /// Reads `n` big-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_int_n(&mut self, nbytes: uint) -> i64;

    /// Reads a little-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_le_uint(&mut self) -> uint;

    /// Reads a little-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_le_int(&mut self) -> int;

    /// Reads a big-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_be_uint(&mut self) -> uint;

    /// Reads a big-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_be_int(&mut self) -> int;

    /// Reads a big-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_be_u64(&mut self) -> u64;

    /// Reads a big-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_be_u32(&mut self) -> u32;

    /// Reads a big-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_be_u16(&mut self) -> u16;

    /// Reads a big-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_be_i64(&mut self) -> i64;

    /// Reads a big-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_be_i32(&mut self) -> i32;

    /// Reads a big-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_be_i16(&mut self) -> i16;

    /// Reads a big-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_be_f64(&mut self) -> f64;

    /// Reads a big-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_be_f32(&mut self) -> f32;

    /// Reads a little-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_le_u64(&mut self) -> u64;

    /// Reads a little-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_le_u32(&mut self) -> u32;

    /// Reads a little-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_le_u16(&mut self) -> u16;

    /// Reads a little-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_le_i64(&mut self) -> i64;

    /// Reads a little-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_le_i32(&mut self) -> i32;

    /// Reads a little-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_le_i16(&mut self) -> i16;

    /// Reads a little-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_le_f64(&mut self) -> f64;

    /// Reads a little-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_le_f32(&mut self) -> f32;

    /// Read a u8.
    ///
    /// `u8`s are 1 byte.
    fn read_u8(&mut self) -> u8;

    /// Read an i8.
    ///
    /// `i8`s are 1 byte.
    fn read_i8(&mut self) -> i8;

}

pub trait WriterByteConversions {
    /// Write the result of passing n through `int::to_str_bytes`.
    fn write_int(&mut self, n: int);

    /// Write the result of passing n through `uint::to_str_bytes`.
    fn write_uint(&mut self, n: uint);

    /// Write a little-endian uint (number of bytes depends on system).
    fn write_le_uint(&mut self, n: uint);

    /// Write a little-endian int (number of bytes depends on system).
    fn write_le_int(&mut self, n: int);

    /// Write a big-endian uint (number of bytes depends on system).
    fn write_be_uint(&mut self, n: uint);

    /// Write a big-endian int (number of bytes depends on system).
    fn write_be_int(&mut self, n: int);

    /// Write a big-endian u64 (8 bytes).
    fn write_be_u64(&mut self, n: u64);

    /// Write a big-endian u32 (4 bytes).
    fn write_be_u32(&mut self, n: u32);

    /// Write a big-endian u16 (2 bytes).
    fn write_be_u16(&mut self, n: u16);

    /// Write a big-endian i64 (8 bytes).
    fn write_be_i64(&mut self, n: i64);

    /// Write a big-endian i32 (4 bytes).
    fn write_be_i32(&mut self, n: i32);

    /// Write a big-endian i16 (2 bytes).
    fn write_be_i16(&mut self, n: i16);

    /// Write a big-endian IEEE754 double-precision floating-point (8 bytes).
    fn write_be_f64(&mut self, f: f64);

    /// Write a big-endian IEEE754 single-precision floating-point (4 bytes).
    fn write_be_f32(&mut self, f: f32);

    /// Write a little-endian u64 (8 bytes).
    fn write_le_u64(&mut self, n: u64);

    /// Write a little-endian u32 (4 bytes).
    fn write_le_u32(&mut self, n: u32);

    /// Write a little-endian u16 (2 bytes).
    fn write_le_u16(&mut self, n: u16);

    /// Write a little-endian i64 (8 bytes).
    fn write_le_i64(&mut self, n: i64);

    /// Write a little-endian i32 (4 bytes).
    fn write_le_i32(&mut self, n: i32);

    /// Write a little-endian i16 (2 bytes).
    fn write_le_i16(&mut self, n: i16);

    /// Write a little-endian IEEE754 double-precision floating-point
    /// (8 bytes).
    fn write_le_f64(&mut self, f: f64);

    /// Write a litten-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    fn write_le_f32(&mut self, f: f32);

    /// Write a u8 (1 byte).
    fn write_u8(&mut self, n: u8);

    /// Write a i8 (1 byte).
    fn write_i8(&mut self, n: i8);
}

#[cfg(test)]
mod test {
    use super::*;
    use option::{Some, None};
    use cell::Cell;
    use rt::io::mem::MemReader;
    use rt::io::mock::*;
    use rt::io::{read_error, placeholder_error};

    #[test]
    fn read_byte() {
        let mut reader = MemReader::new(~[10]);
        let byte = reader.read_byte();
        assert!(byte == Some(10));
    }

    #[test]
    fn read_byte_0_bytes() {
        let mut reader = MockReader::new();
        let count = Cell(0);
        reader.read = |buf| {
            do count.with_mut_ref |count| {
                if *count == 0 {
                    *count = 1;
                    Some(0)
                } else {
                    buf[0] = 10;
                    Some(1)
                }
            }
        };
        let byte = reader.read_byte();
        assert!(byte == Some(10));
    }

    #[test]
    fn read_byte_eof() {
        let mut reader = MockReader::new();
        reader.read = |_| None;
        let byte = reader.read_byte();
        assert!(byte == None);
    }

    #[test]
    fn read_byte_error() {
        let mut reader = MockReader::new();
        reader.read = |_| {
            read_error::cond.raise(placeholder_error());
            None
        };
        do read_error::cond.trap(|_| {
        }).in {
            let byte = reader.read_byte();
            assert!(byte == None);
        }
    }

    #[test]
    fn read_bytes() {
        let mut reader = MemReader::new(~[10, 11, 12, 13]);
        let bytes = reader.read_bytes(4);
        assert!(bytes == ~[10, 11, 12, 13]);
    }

    #[test]
    fn read_bytes_partial() {
        let mut reader = MockReader::new();
        let count = Cell(0);
        reader.read = |buf| {
            do count.with_mut_ref |count| {
                if *count == 0 {
                    *count = 1;
                    buf[0] = 10;
                    buf[1] = 11;
                    Some(2)
                } else {
                    buf[0] = 12;
                    buf[1] = 13;
                    Some(2)
                }
            }
        };
        let bytes = reader.read_bytes(4);
        assert!(bytes == ~[10, 11, 12, 13]);
    }

    #[test]
    fn push_bytes() {
        let mut reader = MemReader::new(~[10, 11, 12, 13]);
        let mut buf = ~[8, 9];
        assert!(reader.push_bytes(&mut buf, 4));
        assert!(buf == ~[8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn push_bytes_partial() {
        let mut reader = MockReader::new();
        let count = Cell(0);
        reader.read = |buf| {
            do count.with_mut_ref |count| {
                if *count == 0 {
                    *count = 1;
                    buf[0] = 10;
                    buf[1] = 11;
                    Some(2)
                } else {
                    buf[0] = 12;
                    buf[1] = 13;
                    Some(2)
                }
            }
        };
        let mut buf = ~[8, 9];
        assert!(reader.push_bytes(&mut buf, 4));
        assert!(buf == ~[8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn push_bytes_eof() {
        let mut reader = MemReader::new(~[10, 11]);
        let mut buf = ~[8, 9];
        assert!(!reader.push_bytes(&mut buf, 4));
        assert!(buf == ~[8, 9, 10, 11]);
    }

    #[test]
    fn push_bytes_error() {
        let mut reader = MockReader::new();
        let count = Cell(0);
        reader.read = |buf| {
            do count.with_mut_ref |count| {
                if *count == 0 {
                    *count = 1;
                    buf[0] = 10;
                    Some(1)
                } else {
                    read_error::cond.raise(placeholder_error());
                    None
                }
            }
        };
        let mut buf = ~[8, 9];
        do read_error::cond.trap(|_| { } ).in {
            assert!(!reader.push_bytes(&mut buf, 4));
        }
        assert!(buf == ~[8, 9, 10]);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn push_bytes_fail_reset_len() {
        use unstable::finally::Finally;

        // push_bytes unsafely sets the vector length. This is testing that
        // upon failure the length is reset correctly.
        let mut reader = MockReader::new();
        let count = Cell(0);
        reader.read = |buf| {
            do count.with_mut_ref |count| {
                if *count == 0 {
                    *count = 1;
                    buf[0] = 10;
                    Some(1)
                } else {
                    read_error::cond.raise(placeholder_error());
                    None
                }
            }
        };
        let buf = @mut ~[8, 9];
        do (|| {
            reader.push_bytes(&mut *buf, 4);
        }).finally {
            // NB: Using rtassert here to trigger abort on failure since this is a should_fail test
            rtassert!(*buf == ~[8, 9, 10]);
        }
    }

}
