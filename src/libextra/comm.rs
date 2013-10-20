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


use std::comm::{GenericChan, GenericSmartChan, GenericPort};
use std::comm::{Chan, Port, Peekable};
use std::comm;

/// An extension of `pipes::stream` that allows both sending and receiving.
pub struct DuplexStream<T, U> {
    priv chan: Chan<T>,
    priv port: Port<U>,
}

// Allow these methods to be used without import:
impl<T:Send,U:Send> DuplexStream<T, U> {
    pub fn send(&self, x: T) {
        self.chan.send(x)
    }
    pub fn try_send(&self, x: T) -> bool {
        self.chan.try_send(x)
    }
    pub fn recv(&self, ) -> U {
        self.port.recv()
    }
    pub fn try_recv(&self) -> Option<U> {
        self.port.try_recv()
    }
    pub fn peek(&self) -> bool {
        self.port.peek()
    }
}

impl<T:Send,U:Send> GenericChan<T> for DuplexStream<T, U> {
    fn send(&self, x: T) {
        self.chan.send(x)
    }
}

impl<T:Send,U:Send> GenericSmartChan<T> for DuplexStream<T, U> {
    fn try_send(&self, x: T) -> bool {
        self.chan.try_send(x)
    }
}

impl<T:Send,U:Send> GenericPort<U> for DuplexStream<T, U> {
    fn recv(&self) -> U {
        self.port.recv()
    }

    fn try_recv(&self) -> Option<U> {
        self.port.try_recv()
    }
}

impl<T:Send,U:Send> Peekable<U> for DuplexStream<T, U> {
    fn peek(&self) -> bool {
        self.port.peek()
    }
}

/// Creates a bidirectional stream.
pub fn DuplexStream<T:Send,U:Send>()
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

/// An extension of `pipes::stream` that provides synchronous message sending.
pub struct SyncChan<T> { priv duplex_stream: DuplexStream<T, ()> }
/// An extension of `pipes::stream` that acknowledges each message received.
pub struct SyncPort<T> { priv duplex_stream: DuplexStream<(), T> }

impl<T: Send> GenericChan<T> for SyncChan<T> {
    fn send(&self, val: T) {
        assert!(self.try_send(val), "SyncChan.send: receiving port closed");
    }
}

impl<T: Send> GenericSmartChan<T> for SyncChan<T> {
    /// Sends a message, or report if the receiver has closed the connection before receiving.
    fn try_send(&self, val: T) -> bool {
        self.duplex_stream.try_send(val) && self.duplex_stream.try_recv().is_some()
    }
}

impl<T: Send> GenericPort<T> for SyncPort<T> {
    fn recv(&self) -> T {
        self.try_recv().expect("SyncPort.recv: sending channel closed")
    }

    fn try_recv(&self) -> Option<T> {
        do self.duplex_stream.try_recv().map |val| {
            self.duplex_stream.try_send(());
            val
        }
    }
}

impl<T: Send> Peekable<T> for SyncPort<T> {
    fn peek(&self) -> bool {
        self.duplex_stream.peek()
    }
}

/// Creates a stream whose channel, upon sending a message, blocks until the message is received.
pub fn rendezvous<T: Send>() -> (SyncPort<T>, SyncChan<T>) {
    let (chan_stream, port_stream) = DuplexStream();
    (SyncPort { duplex_stream: port_stream }, SyncChan { duplex_stream: chan_stream })
}

#[cfg(test)]
mod test {
    use comm::{DuplexStream, rendezvous};
    use std::rt::test::run_in_newsched_task;
    use std::task::spawn_unlinked;


    #[test]
    pub fn DuplexStream1() {
        let (left, right) = DuplexStream();

        left.send(~"abc");
        right.send(123);

        assert!(left.recv() == 123);
        assert!(right.recv() == ~"abc");
    }

    #[test]
    pub fn basic_rendezvous_test() {
        let (port, chan) = rendezvous();

        do spawn {
            chan.send("abc");
        }

        assert!(port.recv() == "abc");
    }

    #[test]
    fn recv_a_lot() {
        // Rendezvous streams should be able to handle any number of messages being sent
        do run_in_newsched_task {
            let (port, chan) = rendezvous();
            do spawn {
                do 1000000.times { chan.send(()) }
            }
            do 1000000.times { port.recv() }
        }
    }

    #[test]
    fn send_and_fail_and_try_recv() {
        let (port, chan) = rendezvous();
        do spawn_unlinked {
            chan.duplex_stream.send(()); // Can't access this field outside this module
            fail!()
        }
        port.recv()
    }

    #[test]
    fn try_send_and_recv_then_fail_before_ack() {
        let (port, chan) = rendezvous();
        do spawn_unlinked {
            port.duplex_stream.recv();
            fail!()
        }
        chan.try_send(());
    }

    #[test]
    #[should_fail]
    fn send_and_recv_then_fail_before_ack() {
        let (port, chan) = rendezvous();
        do spawn_unlinked {
            port.duplex_stream.recv();
            fail!()
        }
        chan.send(());
    }
}
