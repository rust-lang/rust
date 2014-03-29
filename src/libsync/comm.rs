// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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

#![allow(missing_doc)]

use std::comm;

/// An extension of `pipes::stream` that allows both sending and receiving.
pub struct DuplexStream<S, R> {
    priv tx: Sender<S>,
    priv rx: Receiver<R>,
}

/// Creates a bidirectional stream.
pub fn duplex<S: Send, R: Send>() -> (DuplexStream<S, R>, DuplexStream<R, S>) {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    (DuplexStream { tx: tx1, rx: rx2 },
     DuplexStream { tx: tx2, rx: rx1 })
}

// Allow these methods to be used without import:
impl<S:Send,R:Send> DuplexStream<S, R> {
    pub fn send(&self, x: S) {
        self.tx.send(x)
    }
    pub fn try_send(&self, x: S) -> bool {
        self.tx.try_send(x)
    }
    pub fn recv(&self) -> R {
        self.rx.recv()
    }
    pub fn try_recv(&self) -> comm::TryRecvResult<R> {
        self.rx.try_recv()
    }
    pub fn recv_opt(&self) -> Option<R> {
        self.rx.recv_opt()
    }
}

#[cfg(test)]
mod test {
    use comm::{duplex};


    #[test]
    pub fn DuplexStream1() {
        let (left, right) = duplex();

        left.send(~"abc");
        right.send(123);

        assert!(left.recv() == 123);
        assert!(right.recv() == ~"abc");
    }
}
