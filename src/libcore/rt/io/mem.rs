// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Readers and Writers for in-memory buffers
//!
//! # XXX
//!
//! * Should probably have something like this for strings.
//! * Should they implement Closable? Would take extra state.

use prelude::*;
use super::*;


/// Writes to an owned, growable byte vector
pub struct MemWriter {
    buf: ~[u8]
}

impl MemWriter {
    pub fn new() -> MemWriter { MemWriter { buf: ~[] } }
}

impl Writer for MemWriter {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { /* no-op */ }
}

impl Seek for MemWriter {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

impl Decorator<~[u8]> for MemWriter {

    fn inner(self) -> ~[u8] {
        match self {
            MemWriter { buf: buf } => buf
        }
    }

    fn inner_ref<'a>(&'a self) -> &'a ~[u8] {
        match *self {
            MemWriter { buf: ref buf } => buf
        }
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut ~[u8] {
        match *self {
            MemWriter { buf: ref mut buf } => buf
        }
    }
}

/// Reads from an owned byte vector
pub struct MemReader {
    buf: ~[u8],
    pos: uint
}

impl MemReader {
    pub fn new(buf: ~[u8]) -> MemReader {
        MemReader {
            buf: buf,
            pos: 0
        }
    }
}

impl Reader for MemReader {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl Seek for MemReader {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

impl Decorator<~[u8]> for MemReader {

    fn inner(self) -> ~[u8] {
        match self {
            MemReader { buf: buf, _ } => buf
        }
    }

    fn inner_ref<'a>(&'a self) -> &'a ~[u8] {
        match *self {
            MemReader { buf: ref buf, _ } => buf
        }
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut ~[u8] {
        match *self {
            MemReader { buf: ref mut buf, _ } => buf
        }
    }
}


/// Writes to a fixed-size byte slice
struct BufWriter<'self> {
    buf: &'self mut [u8],
    pos: uint
}

impl<'self> BufWriter<'self> {
    pub fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a> {
        BufWriter {
            buf: buf,
            pos: 0
        }
    }
}

impl<'self> Writer for BufWriter<'self> {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl<'self> Seek for BufWriter<'self> {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}


/// Reads from a fixed-size byte slice
struct BufReader<'self> {
    buf: &'self [u8],
    pos: uint
}

impl<'self> BufReader<'self> {
    pub fn new<'a>(buf: &'a [u8]) -> BufReader<'a> {
        BufReader {
            buf: buf,
            pos: 0
        }
    }
}

impl<'self> Reader for BufReader<'self> {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl<'self> Seek for BufReader<'self> {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}