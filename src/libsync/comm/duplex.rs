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
#![allow(deprecated)]
#![deprecated = "This type is replaced by having a pair of channels. This type \
                 is not fully composable with other channels in terms of \
                 or possible semantics on a duplex stream. It will be removed \
                 soon"]

use core::prelude::*;

use comm;
use comm::{Sender, Receiver, channel};

/// An extension of `pipes::stream` that allows both sending and receiving.
pub struct DuplexStream<S, R> {
    tx: Sender<S>,
    rx: Receiver<R>,
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
    pub fn send_opt(&self, x: S) -> Result<(), S> {
        self.tx.send_opt(x)
    }
    pub fn recv(&self) -> R {
        self.rx.recv()
    }
    pub fn try_recv(&self) -> Result<R, comm::TryRecvError> {
        self.rx.try_recv()
    }
    pub fn recv_opt(&self) -> Result<R, ()> {
        self.rx.recv_opt()
    }
}

#[cfg(test)]
mod test {
    use std::prelude::*;
    use comm::{duplex};

    #[test]
    pub fn duplex_stream_1() {
        let (left, right) = duplex();

        left.send("abc".to_string());
        right.send(123i);

        assert!(left.recv() == 123);
        assert!(right.recv() == "abc".to_string());
    }
}
