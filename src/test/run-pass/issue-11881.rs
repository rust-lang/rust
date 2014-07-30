// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rbml;
extern crate serialize;

use std::io;
use std::io::{IoError, IoResult, SeekStyle};
use std::slice;

use serialize::{Encodable, Encoder};
use serialize::json;
use rbml::writer;

static BUF_CAPACITY: uint = 128;

fn combine(seek: SeekStyle, cur: uint, end: uint, offset: i64) -> IoResult<u64> {
    // compute offset as signed and clamp to prevent overflow
    let pos = match seek {
        io::SeekSet => 0,
        io::SeekEnd => end,
        io::SeekCur => cur,
    } as i64;

    if offset + pos < 0 {
        Err(IoError {
            kind: io::InvalidInput,
            desc: "invalid seek to a negative offset",
            detail: None
        })
    } else {
        Ok((offset + pos) as u64)
    }
}

/// Writes to an owned, growable byte vector that supports seeking.
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::SeekableMemWriter;
///
/// let mut w = SeekableMemWriter::new();
/// w.write([0, 1, 2]);
///
/// assert_eq!(w.unwrap(), vec!(0, 1, 2));
/// ```
pub struct SeekableMemWriter {
    buf: Vec<u8>,
    pos: uint,
}

impl SeekableMemWriter {
    /// Create a new `SeekableMemWriter`.
    #[inline]
    pub fn new() -> SeekableMemWriter {
        SeekableMemWriter::with_capacity(BUF_CAPACITY)
    }
    /// Create a new `SeekableMemWriter`, allocating at least `n` bytes for
    /// the internal buffer.
    #[inline]
    pub fn with_capacity(n: uint) -> SeekableMemWriter {
        SeekableMemWriter { buf: Vec::with_capacity(n), pos: 0 }
    }

    /// Acquires an immutable reference to the underlying buffer of this
    /// `SeekableMemWriter`.
    ///
    /// No method is exposed for acquiring a mutable reference to the buffer
    /// because it could corrupt the state of this `MemWriter`.
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a [u8] { self.buf.as_slice() }

    /// Unwraps this `SeekableMemWriter`, returning the underlying buffer
    #[inline]
    pub fn unwrap(self) -> Vec<u8> { self.buf }
}

impl Writer for SeekableMemWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        if self.pos == self.buf.len() {
            self.buf.push_all(buf)
        } else {
            // Make sure the internal buffer is as least as big as where we
            // currently are
            let difference = self.pos as i64 - self.buf.len() as i64;
            if difference > 0 {
                self.buf.grow(difference as uint, &0);
            }

            // Figure out what bytes will be used to overwrite what's currently
            // there (left), and what will be appended on the end (right)
            let cap = self.buf.len() - self.pos;
            let (left, right) = if cap <= buf.len() {
                (buf.slice_to(cap), buf.slice_from(cap))
            } else {
                (buf, &[])
            };

            // Do the necessary writes
            if left.len() > 0 {
                slice::bytes::copy_memory(self.buf.mut_slice_from(self.pos), left);
            }
            if right.len() > 0 {
                self.buf.push_all(right);
            }
        }

        // Bump us forward
        self.pos += buf.len();
        Ok(())
    }
}

impl Seek for SeekableMemWriter {
    #[inline]
    fn tell(&self) -> IoResult<u64> { Ok(self.pos as u64) }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> {
        let new = try!(combine(style, self.pos, self.buf.len(), pos));
        self.pos = new as uint;
        Ok(())
    }
}

#[deriving(Encodable)]
struct Foo {
    baz: bool,
}

#[deriving(Encodable)]
struct Bar {
    froboz: uint,
}

enum WireProtocol {
    JSON,
    RBML,
    // ...
}

fn encode_json<'a,
               T: Encodable<json::Encoder<'a>,
                            std::io::IoError>>(val: &T,
                                               wr: &'a mut SeekableMemWriter) {
    let mut encoder = json::Encoder::new(wr);
    val.encode(&mut encoder);
}
fn encode_rbml<'a,
               T: Encodable<writer::Encoder<'a, SeekableMemWriter>,
                            std::io::IoError>>(val: &T,
                                               wr: &'a mut SeekableMemWriter) {
    let mut encoder = writer::Encoder::new(wr);
    val.encode(&mut encoder);
}

pub fn main() {
    let target = Foo{baz: false,};
    let mut wr = SeekableMemWriter::new();
    let proto = JSON;
    match proto {
        JSON => encode_json(&target, &mut wr),
        RBML => encode_rbml(&target, &mut wr)
    }
}
