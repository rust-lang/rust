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

use core::prelude::*;

use comm;
use comm::{Sender, Receiver, channel};

/// An extension of `pipes::stream` that allows both sending and receiving
/// data over two channels
pub struct DuplexStream<S, R> {
    tx: Sender<S>,
    rx: Receiver<R>,
}

/// Creates a bidirectional stream.
///
/// # Example
/// ```
/// use std::comm;
///
/// let (left, right) = comm::duplex();
///
/// left.send(("ABC").to_string());
/// right.send(123);
///
/// assert!(left.recv() == 123);
/// assert!(right.recv() == "ABC".to_string());
/// ```
pub fn duplex<S: Send, R: Send>() -> (DuplexStream<S, R>, DuplexStream<R, S>) {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    (DuplexStream { tx: tx1, rx: rx2 },
     DuplexStream { tx: tx2, rx: rx1 })
}

// Allow these methods to be used without import:
impl<S:Send,R:Send> DuplexStream<S, R> {
    /// Sends a value along this duplex to be received by the corresponding
    /// receiver.
    ///
    /// Rust duplexes are infinitely buffered so this method will never block.
    ///
    /// # Failure
    ///
    /// This function will fail if the other end of the duplex has hung up.
    /// This means that if the corresponding receiver has fallen out of scope,
    /// this function will trigger a fail message saying that a message is being
    /// sent on a closed duplex.
    ///
    /// Note that if this function does not fail, it does not mean that the data
    /// will be successfully received. All sends are placed into a queue, so it
    /// is possible for a send to succeed (the other end is alive), but then the
    /// other end could immediately disconnect.
    ///
    /// The purpose of this functionality is to propagate failure among tasks.
    /// If failure is not desired, then consider using the send_opt method.
    pub fn send(&self, x: S) {
        self.tx.send(x)
    }

    /// Optionally send data to the channel.
    ///
    /// # Example
    /// ```
    /// use std::comm;
    ///
    /// let (left, right) = comm::duplex();
    ///
    /// left.send("ABC".to_string());
    /// right.send(123);
    /// assert!(right.recv() == "ABC".to_string());
    /// drop(right);
    /// assert!(left.recv() == 123);
    /// assert_eq!(left.send_opt("ABC".to_string()), Err("ABC".to_string()));
    /// ```
    pub fn send_opt(&self, x: S) -> Result<(), S> {
        self.tx.send_opt(x)
    }

    /// Receive data from the channel.
    pub fn recv(&self) -> R {
        self.rx.recv()
    }

    /// Try to receive data from the channel.
    ///
    /// # Example
    /// ```
    /// use std::comm;
    ///
    /// let (left, right) = comm::duplex();
    /// let a = "ABC".to_string();
    /// let b:u32 = 123;
    ///
    /// left.send(a.clone());
    /// assert_eq!(right.recv(), a);
    /// right.send(b);
    /// assert_eq!(left.recv(), b);
    /// // Here the channel is empty so it return an error.
    /// assert_eq!(left.try_recv(), Err(comm::Empty));
    /// ```
}
    pub fn try_recv(&self) -> Result<R, comm::TryRecvError> {
        self.rx.try_recv()
    }

    /// Optionally receive data from the channel.
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
