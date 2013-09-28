// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use super::super::*;
use super::super::support::PathLike;

pub struct UnixStream;

impl UnixStream {
    pub fn connect<P: PathLike>(_path: &P) -> Option<UnixStream> {
        fail2!()
    }
}

impl Reader for UnixStream {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail2!() }

    fn eof(&mut self) -> bool { fail2!() }
}

impl Writer for UnixStream {
    fn write(&mut self, _v: &[u8]) { fail2!() }

    fn flush(&mut self) { fail2!() }
}

pub struct UnixListener;

impl UnixListener {
    pub fn bind<P: PathLike>(_path: &P) -> Option<UnixListener> {
        fail2!()
    }
}

impl Listener<UnixStream, UnixAcceptor> for UnixListener {
    fn listen(self) -> Option<UnixAcceptor> { fail2!() }
}

pub struct UnixAcceptor;

impl Acceptor<UnixStream> for UnixAcceptor {
    fn accept(&mut self) -> Option<UnixStream> { fail2!() }
}
