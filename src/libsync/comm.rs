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

#[allow(missing_doc)];

use std::comm;

/// An extension of `pipes::stream` that allows both sending and receiving.
pub struct DuplexStream<T, U> {
    priv tx: Sender<T>,
    priv rx: Receiver<U>,
}

/// Creates a bidirectional stream.
pub fn duplex<T: Send, U: Send>() -> (DuplexStream<T, U>, DuplexStream<U, T>) {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    (DuplexStream { tx: tx1, rx: rx2 },
     DuplexStream { tx: tx2, rx: rx1 })
}

// Allow these methods to be used without import:
impl<T:Send,U:Send> DuplexStream<T, U> {
    pub fn send(&self, x: T) {
        self.tx.send(x)
    }
    pub fn try_send(&self, x: T) -> bool {
        self.tx.try_send(x)
    }
    pub fn recv(&self) -> U {
        self.rx.recv()
    }
    pub fn try_recv(&self) -> comm::TryRecvResult<U> {
        self.rx.try_recv()
    }
    pub fn recv_opt(&self) -> Option<U> {
        self.rx.recv_opt()
    }
}

/// An extension of `pipes::stream` that provides synchronous message sending.
pub struct SyncSender<T> { priv duplex_stream: DuplexStream<T, ()> }
/// An extension of `pipes::stream` that acknowledges each message received.
pub struct SyncReceiver<T> { priv duplex_stream: DuplexStream<(), T> }

impl<T: Send> SyncSender<T> {
    pub fn send(&self, val: T) {
        assert!(self.try_send(val), "SyncSender.send: receiving port closed");
    }

    /// Sends a message, or report if the receiver has closed the connection
    /// before receiving.
    pub fn try_send(&self, val: T) -> bool {
        self.duplex_stream.try_send(val) && self.duplex_stream.recv_opt().is_some()
    }
}

impl<T: Send> SyncReceiver<T> {
    pub fn recv(&self) -> T {
        self.recv_opt().expect("SyncReceiver.recv: sending channel closed")
    }

    pub fn recv_opt(&self) -> Option<T> {
        self.duplex_stream.recv_opt().map(|val| {
            self.duplex_stream.try_send(());
            val
        })
    }

    pub fn try_recv(&self) -> comm::TryRecvResult<T> {
        match self.duplex_stream.try_recv() {
            comm::Data(t) => { self.duplex_stream.try_send(()); comm::Data(t) }
            state => state,
        }
    }
}

/// Creates a stream whose channel, upon sending a message, blocks until the
/// message is received.
pub fn rendezvous<T: Send>() -> (SyncReceiver<T>, SyncSender<T>) {
    let (chan_stream, port_stream) = duplex();
    (SyncReceiver { duplex_stream: port_stream },
     SyncSender { duplex_stream: chan_stream })
}

#[cfg(test)]
mod test {
    use comm::{duplex, rendezvous};


    #[test]
    pub fn DuplexStream1() {
        let (left, right) = duplex();

        left.send(~"abc");
        right.send(123);

        assert!(left.recv() == 123);
        assert!(right.recv() == ~"abc");
    }

    #[test]
    pub fn basic_rendezvous_test() {
        let (port, chan) = rendezvous();

        spawn(proc() {
            chan.send("abc");
        });

        assert!(port.recv() == "abc");
    }

    #[test]
    fn recv_a_lot() {
        // Rendezvous streams should be able to handle any number of messages being sent
        let (port, chan) = rendezvous();
        spawn(proc() {
            for _ in range(0, 10000) { chan.send(()); }
        });
        for _ in range(0, 10000) { port.recv(); }
    }

    #[test]
    fn send_and_fail_and_try_recv() {
        let (port, chan) = rendezvous();
        spawn(proc() {
            chan.duplex_stream.send(()); // Can't access this field outside this module
            fail!()
        });
        port.recv()
    }

    #[test]
    fn try_send_and_recv_then_fail_before_ack() {
        let (port, chan) = rendezvous();
        spawn(proc() {
            port.duplex_stream.recv();
            fail!()
        });
        chan.try_send(());
    }

    #[test]
    #[should_fail]
    fn send_and_recv_then_fail_before_ack() {
        let (port, chan) = rendezvous();
        spawn(proc() {
            port.duplex_stream.recv();
            fail!()
        });
        chan.send(());
    }
}
