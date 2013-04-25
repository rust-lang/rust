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
// XXX: Iteration should probably be considered seperately

pub trait ReaderUtil {

    /// Reads `len` bytes and gives you back a new vector
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns an empty
    /// vector if the condition is handled.
    fn read_bytes(&mut self, len: uint) -> ~[u8];

    /// Reads all remaining bytes from the stream.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns an empty
    /// vector if the condition is handled.
    fn read_to_end(&mut self) -> ~[u8];

}

pub trait ReaderByteConversions {
    /// Reads `n` little-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_uint_n(&mut self, nbytes: uint) -> u64;

    /// Reads `n` little-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_int_n(&mut self, nbytes: uint) -> i64;

    /// Reads `n` big-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_uint_n(&mut self, nbytes: uint) -> u64;

    /// Reads `n` big-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_int_n(&mut self, nbytes: uint) -> i64;

    /// Reads a little-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependant.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_uint(&mut self) -> uint;

    /// Reads a little-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_int(&mut self) -> int;

    /// Reads a big-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependant.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_uint(&mut self) -> uint;

    /// Reads a big-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_int(&mut self) -> int;

    /// Reads a big-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_u64(&mut self) -> u64;

    /// Reads a big-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_u32(&mut self) -> u32;

    /// Reads a big-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_u16(&mut self) -> u16;

    /// Reads a big-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_i64(&mut self) -> i64;

    /// Reads a big-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_i32(&mut self) -> i32;

    /// Reads a big-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_i16(&mut self) -> i16;

    /// Reads a big-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_f64(&mut self) -> f64;

    /// Reads a big-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_be_f32(&mut self) -> f32;

    /// Reads a little-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_u64(&mut self) -> u64;

    /// Reads a little-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_u32(&mut self) -> u32;

    /// Reads a little-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_u16(&mut self) -> u16;

    /// Reads a little-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_i64(&mut self) -> i64;

    /// Reads a little-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_i32(&mut self) -> i32;

    /// Reads a little-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_i16(&mut self) -> i16;

    /// Reads a little-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_f64(&mut self) -> f64;

    /// Reads a little-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_le_f32(&mut self) -> f32;

    /// Read a u8.
    ///
    /// `u8`s are 1 byte.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_u8(&mut self) -> u8;

    /// Read an i8.
    ///
    /// `i8`s are 1 byte.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. Returns `0` if
    /// the condition is handled.
    fn read_i8(&mut self) -> i8;

}

pub trait WriterByteConversions {
    /// Write the result of passing n through `int::to_str_bytes`.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_int(&mut self, n: int);

    /// Write the result of passing n through `uint::to_str_bytes`.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_uint(&mut self, n: uint);

    /// Write a little-endian uint (number of bytes depends on system).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_uint(&mut self, n: uint);

    /// Write a little-endian int (number of bytes depends on system).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_int(&mut self, n: int);

    /// Write a big-endian uint (number of bytes depends on system).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_uint(&mut self, n: uint);

    /// Write a big-endian int (number of bytes depends on system).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_int(&mut self, n: int);

    /// Write a big-endian u64 (8 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_u64(&mut self, n: u64);

    /// Write a big-endian u32 (4 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_u32(&mut self, n: u32);

    /// Write a big-endian u16 (2 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_u16(&mut self, n: u16);

    /// Write a big-endian i64 (8 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_i64(&mut self, n: i64);

    /// Write a big-endian i32 (4 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_i32(&mut self, n: i32);

    /// Write a big-endian i16 (2 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_i16(&mut self, n: i16);

    /// Write a big-endian IEEE754 double-precision floating-point (8 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_f64(&mut self, f: f64);

    /// Write a big-endian IEEE754 single-precision floating-point (4 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_be_f32(&mut self, f: f32);

    /// Write a little-endian u64 (8 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_u64(&mut self, n: u64);

    /// Write a little-endian u32 (4 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_u32(&mut self, n: u32);

    /// Write a little-endian u16 (2 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_u16(&mut self, n: u16);

    /// Write a little-endian i64 (8 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_i64(&mut self, n: i64);

    /// Write a little-endian i32 (4 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_i32(&mut self, n: i32);

    /// Write a little-endian i16 (2 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_i16(&mut self, n: i16);

    /// Write a little-endian IEEE754 double-precision floating-point
    /// (8 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_f64(&mut self, f: f64);

    /// Write a litten-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_le_f32(&mut self, f: f32);

    /// Write a u8 (1 byte).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_u8(&mut self, n: u8);

    /// Write a i8 (1 byte).
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error.
    fn write_i8(&mut self, n: i8);
}
