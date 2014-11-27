// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementations of I/O traits for the IoResult type
//!
//! I/O constructors return option types to allow errors to be handled.
//! These implementations allow e.g. `IoResult<File>` to be used
//! as a `Reader` without unwrapping the result first.

use clone::Clone;
use result::{Ok, Err};
use super::{Reader, Writer, Listener, Acceptor, Seek, SeekStyle, IoResult};

impl<W: Writer> Writer for IoResult<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    fn flush(&mut self) -> IoResult<()> { unimplemented!() }
}

impl<R: Reader> Reader for IoResult<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl<S: Seek> Seek for IoResult<S> {
    fn tell(&self) -> IoResult<u64> { unimplemented!() }
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> { unimplemented!() }
}

impl<T, A: Acceptor<T>, L: Listener<T, A>> Listener<T, A> for IoResult<L> {
    fn listen(self) -> IoResult<A> { unimplemented!() }
}

impl<T, A: Acceptor<T>> Acceptor<T> for IoResult<A> {
    fn accept(&mut self) -> IoResult<T> { unimplemented!() }
}
