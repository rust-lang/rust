// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::Option;
use comm::{GenericPort, GenericChan};
use super::{Reader, Writer};

struct PortReader<P>;

impl<P: GenericPort<~[u8]>> PortReader<P> {
    pub fn new(_port: P) -> PortReader<P> { fail!() }
}

impl<P: GenericPort<~[u8]>> Reader for PortReader<P> {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

struct ChanWriter<C>;

impl<C: GenericChan<~[u8]>> ChanWriter<C> {
    pub fn new(_chan: C) -> ChanWriter<C> { fail!() }
}

impl<C: GenericChan<~[u8]>> Writer for ChanWriter<C> {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

struct ReaderPort<R>;

impl<R: Reader> ReaderPort<R> {
    pub fn new(_reader: R) -> ReaderPort<R> { fail!() }
}

impl<R: Reader> GenericPort<~[u8]> for ReaderPort<R> {
    fn recv(&self) -> ~[u8] { fail!() }

    fn try_recv(&self) -> Option<~[u8]> { fail!() }
}

struct WriterChan<W>;

impl<W: Writer> WriterChan<W> {
    pub fn new(_writer: W) -> WriterChan<W> { fail!() }
}

impl<W: Writer> GenericChan<~[u8]> for WriterChan<W> {
    fn send(&self, _x: ~[u8]) { fail!() }
}
