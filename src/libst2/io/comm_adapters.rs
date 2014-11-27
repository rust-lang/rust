// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clone::Clone;
use cmp;
use comm::{Sender, Receiver};
use io;
use option::{None, Some};
use result::{Ok, Err};
use slice::{bytes, CloneSliceAllocPrelude, SlicePrelude};
use super::{Buffer, Reader, Writer, IoResult};
use vec::Vec;

/// Allows reading from a rx.
///
/// # Example
///
/// ```
/// use std::io::ChanReader;
///
/// let (tx, rx) = channel();
/// # drop(tx);
/// let mut reader = ChanReader::new(rx);
///
/// let mut buf = [0u8, ..100];
/// match reader.read(&mut buf) {
///     Ok(nread) => println!("Read {} bytes", nread),
///     Err(e) => println!("read error: {}", e),
/// }
/// ```
pub struct ChanReader {
    buf: Vec<u8>,          // A buffer of bytes received but not consumed.
    pos: uint,             // How many of the buffered bytes have already be consumed.
    rx: Receiver<Vec<u8>>, // The Receiver to pull data from.
    closed: bool,          // Whether the channel this Receiver connects to has been closed.
}

impl ChanReader {
    /// Wraps a `Port` in a `ChanReader` structure
    pub fn new(rx: Receiver<Vec<u8>>) -> ChanReader { unimplemented!() }
}

impl Buffer for ChanReader {
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { unimplemented!() }

    fn consume(&mut self, amt: uint) { unimplemented!() }
}

impl Reader for ChanReader {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

/// Allows writing to a tx.
///
/// # Example
///
/// ```
/// # #![allow(unused_must_use)]
/// use std::io::ChanWriter;
///
/// let (tx, rx) = channel();
/// # drop(rx);
/// let mut writer = ChanWriter::new(tx);
/// writer.write("hello, world".as_bytes());
/// ```
pub struct ChanWriter {
    tx: Sender<Vec<u8>>,
}

impl ChanWriter {
    /// Wraps a channel in a `ChanWriter` structure
    pub fn new(tx: Sender<Vec<u8>>) -> ChanWriter { unimplemented!() }
}

impl Clone for ChanWriter {
    fn clone(&self) -> ChanWriter { unimplemented!() }
}

impl Writer for ChanWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
}
