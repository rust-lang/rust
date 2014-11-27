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

#![allow(missing_docs)]

// FIXME: Not sure how this should be structured
// FIXME: Iteration should probably be considered separately

use io::{IoError, IoResult, Reader};
use io;
use iter::Iterator;
use num::Int;
use option::{Option, Some, None};
use ptr::RawPtr;
use result::{Ok, Err};
use slice::{SlicePrelude, AsSlice};

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
pub struct Bytes<'r, T:'r> {
    reader: &'r mut T,
}

impl<'r, R: Reader> Bytes<'r, R> {
    /// Constructs a new byte iterator from the given Reader instance.
    pub fn new(r: &'r mut R) -> Bytes<'r, R> { unimplemented!() }
}

impl<'r, R: Reader> Iterator<IoResult<u8>> for Bytes<'r, R> {
    #[inline]
    fn next(&mut self) -> Option<IoResult<u8>> { unimplemented!() }
}

/// Converts an 8-bit to 64-bit unsigned value to a little-endian byte
/// representation of the given size. If the size is not big enough to
/// represent the value, then the high-order bytes are truncated.
///
/// Arguments:
///
/// * `n`: The value to convert.
/// * `size`: The size of the value, in bytes. This must be 8 or less, or task
///           panic occurs. If this is less than 8, then a value of that
///           many bytes is produced. For example, if `size` is 4, then a
///           32-bit byte representation is produced.
/// * `f`: A callback that receives the value.
///
/// This function returns the value returned by the callback, for convenience.
pub fn u64_to_le_bytes<T>(n: u64, size: uint, f: |v: &[u8]| -> T) -> T { unimplemented!() }

/// Converts an 8-bit to 64-bit unsigned value to a big-endian byte
/// representation of the given size. If the size is not big enough to
/// represent the value, then the high-order bytes are truncated.
///
/// Arguments:
///
/// * `n`: The value to convert.
/// * `size`: The size of the value, in bytes. This must be 8 or less, or task
///           panic occurs. If this is less than 8, then a value of that
///           many bytes is produced. For example, if `size` is 4, then a
///           32-bit byte representation is produced.
/// * `f`: A callback that receives the value.
///
/// This function returns the value returned by the callback, for convenience.
pub fn u64_to_be_bytes<T>(n: u64, size: uint, f: |v: &[u8]| -> T) -> T { unimplemented!() }

/// Extracts an 8-bit to 64-bit unsigned big-endian value from the given byte
/// buffer and returns it as a 64-bit value.
///
/// Arguments:
///
/// * `data`: The buffer in which to extract the value.
/// * `start`: The offset at which to extract the value.
/// * `size`: The size of the value in bytes to extract. This must be 8 or
///           less, or task panic occurs. If this is less than 8, then only
///           that many bytes are parsed. For example, if `size` is 4, then a
///           32-bit value is parsed.
pub fn u64_from_be_bytes(data: &[u8], start: uint, size: uint) -> u64 { unimplemented!() }
