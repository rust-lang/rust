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

/// An extension of `pipes::stream` that allows both sending and receiving.
pub struct DuplexStream<T, U> {
    priv chan: Chan<T>,
    priv port: Port<U>,
}

// Allow these methods to be used without import:
impl<T:Send,U:Send> DuplexStream<T, U> {
    /// Creates a bidirectional stream.
    pub fn new() -> (DuplexStream<T, U>, DuplexStream<U, T>) {
        let (p1, c2) = Chan::new();
        let (p2, c1) = Chan::new();
        (DuplexStream { chan: c1, port: p1 },
         DuplexStream { chan: c2, port: p2 })
    }
    pub fn send(&self, x: T) {
        self.chan.send(x)
    }
    pub fn try_send(&self, x: T) -> bool {
        self.chan.try_send(x)
    }
    pub fn recv(&self) -> U {
        self.port.recv()
    }
    pub fn try_recv(&self) -> Option<U> {
        self.port.try_recv()
    }
    pub fn recv_opt(&self) -> Option<U> {
        self.port.recv_opt()
    }
}

/// An extension of `pipes::stream` that provides synchronous message sending.
pub struct SyncChan<T> { priv duplex_stream: DuplexStream<T, ()> }
/// An extension of `pipes::stream` that acknowledges each message received.
pub struct SyncPort<T> { priv duplex_stream: DuplexStream<(), T> }

impl<T: Send> SyncChan<T> {
    pub fn send(&self, val: T) {
        assert!(self.try_send(val), "SyncChan.send: receiving port closed");
    }

    /// Sends a message, or report if the receiver has closed the connection
    /// before receiving.
    pub fn try_send(&self, val: T) -> bool {
        self.duplex_stream.try_send(val) && self.duplex_stream.recv_opt().is_some()
    }
}

impl<T: Send> SyncPort<T> {
    pub fn recv(&self) -> T {
        self.recv_opt().expect("SyncPort.recv: sending channel closed")
    }

    pub fn recv_opt(&self) -> Option<T> {
        self.duplex_stream.recv_opt().map(|val| {
            self.duplex_stream.try_send(());
            val
        })
    }

    pub fn try_recv(&self) -> Option<T> {
        self.duplex_stream.try_recv().map(|val| {
            self.duplex_stream.try_send(());
            val
        })
    }
}

/// Creates a stream whose channel, upon sending a message, blocks until the
/// message is received.
pub fn rendezvous<T: Send>() -> (SyncPort<T>, SyncChan<T>) {
    let (chan_stream, port_stream) = DuplexStream::new();
    (SyncPort { duplex_stream: port_stream },
     SyncChan { duplex_stream: chan_stream })
}

#[cfg(test)]
mod test {
    use comm::{DuplexStream, rendezvous};
    use std::rt::test::run_in_uv_task;


    #[test]
    pub fn DuplexStream1() {
        let (mut left, mut right) = DuplexStream::new();

        left.send(~"abc");
        right.send(123);

        assert!(left.recv() == 123);
        assert!(right.recv() == ~"abc");
    }

    #[test]
    pub fn basic_rendezvous_test() {
        let (mut port, chan) = rendezvous();

        do spawn {
            let mut chan = chan;
            chan.send("abc");
        }

        assert!(port.recv() == "abc");
    }

    #[test]
    fn recv_a_lot() {
        // Rendezvous streams should be able to handle any number of messages being sent
        do run_in_uv_task {
            let (mut port, chan) = rendezvous();
            do spawn {
                let mut chan = chan;
                1000000.times(|| { chan.send(()) })
            }
            1000000.times(|| { port.recv() })
        }
    }

    #[test]
    fn send_and_fail_and_try_recv() {
        let (mut port, chan) = rendezvous();
        do spawn {
            let mut chan = chan;
            chan.duplex_stream.send(()); // Can't access this field outside this module
            fail!()
        }
        port.recv()
    }

    #[test]
    fn try_send_and_recv_then_fail_before_ack() {
        let (port, mut chan) = rendezvous();
        do spawn {
            let mut port = port;
            port.duplex_stream.recv();
            fail!()
        }
        chan.try_send(());
    }

    #[test]
    #[should_fail]
    fn send_and_recv_then_fail_before_ack() {
        let (port, mut chan) = rendezvous();
        do spawn {
            let mut port = port;
            port.duplex_stream.recv();
            fail!()
        }
        chan.send(());
    }
}
