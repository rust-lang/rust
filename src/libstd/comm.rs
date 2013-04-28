// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Higher level communication abstractions.

*/

use core::comm::{GenericChan, GenericSmartChan, GenericPort};
use core::comm::{Chan, Port, Selectable, Peekable};
use core::pipes;

/// An extension of `pipes::stream` that allows both sending and receiving.
pub struct DuplexStream<T, U> {
    priv chan: Chan<T>,
    priv port: Port<U>,
}

// Allow these methods to be used without import:
pub impl<T:Owned,U:Owned> DuplexStream<T, U> {
    fn send(&self, x: T) {
        self.chan.send(x)
    }
    fn try_send(&self, x: T) -> bool {
        self.chan.try_send(x)
    }
    fn recv(&self, ) -> U {
        self.port.recv()
    }
    fn try_recv(&self) -> Option<U> {
        self.port.try_recv()
    }
    fn peek(&self) -> bool {
        self.port.peek()
    }
}

impl<T:Owned,U:Owned> GenericChan<T> for DuplexStream<T, U> {
    fn send(&self, x: T) {
        self.chan.send(x)
    }
}

impl<T:Owned,U:Owned> GenericSmartChan<T> for DuplexStream<T, U> {
    fn try_send(&self, x: T) -> bool {
        self.chan.try_send(x)
    }
}

impl<T:Owned,U:Owned> GenericPort<U> for DuplexStream<T, U> {
    fn recv(&self) -> U {
        self.port.recv()
    }

    fn try_recv(&self) -> Option<U> {
        self.port.try_recv()
    }
}

impl<T:Owned,U:Owned> Peekable<U> for DuplexStream<T, U> {
    fn peek(&self) -> bool {
        self.port.peek()
    }
}

impl<T:Owned,U:Owned> Selectable for DuplexStream<T, U> {
    fn header(&self) -> *pipes::PacketHeader {
        self.port.header()
    }
}

/// Creates a bidirectional stream.
pub fn DuplexStream<T:Owned,U:Owned>()
    -> (DuplexStream<T, U>, DuplexStream<U, T>)
{
    let (p1, c2) = comm::stream();
    let (p2, c1) = comm::stream();
    (DuplexStream {
        chan: c1,
        port: p1
    },
     DuplexStream {
         chan: c2,
         port: p2
     })
}

#[cfg(test)]
mod test {
    use comm::DuplexStream;

    #[test]
    pub fn DuplexStream1() {
        let (left, right) = DuplexStream();

        left.send(~"abc");
        right.send(123);

        assert!(left.recv() == 123);
        assert!(right.recv() == ~"abc");
    }
}
