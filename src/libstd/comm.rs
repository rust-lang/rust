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
Message passing
*/

#[allow(missing_doc)];

use clone::Clone;
use iter::Iterator;
use kinds::Send;
use option::Option;
use rtcomm = rt::comm;

/// A trait for things that can send multiple messages.
pub trait GenericChan<T> {
    /// Sends a message.
    fn send(&self, x: T);
}

/// Things that can send multiple messages and can detect when the receiver
/// is closed
pub trait GenericSmartChan<T> {
    /// Sends a message, or report if the receiver has closed the connection.
    fn try_send(&self, x: T) -> bool;
}

/// Trait for non-rescheduling send operations, similar to `send_deferred` on ChanOne.
pub trait SendDeferred<T> {
    fn send_deferred(&self, val: T);
    fn try_send_deferred(&self, val: T) -> bool;
}

/// A trait for things that can receive multiple messages.
pub trait GenericPort<T> {
    /// Receives a message, or fails if the connection closes.
    fn recv(&self) -> T;

    /// Receives a message, or returns `none` if
    /// the connection is closed or closes.
    fn try_recv(&self) -> Option<T>;

    /// Returns an iterator that breaks once the connection closes.
    ///
    /// # Example
    ///
    /// ~~~rust
    /// do spawn {
    ///     for x in port.recv_iter() {
    ///         if pred(x) { break; }
    ///         println!("{}", x);
    ///     }
    /// }
    /// ~~~
    fn recv_iter<'a>(&'a self) -> RecvIterator<'a, Self> {
        RecvIterator { port: self }
    }
}

pub struct RecvIterator<'a, P> {
    priv port: &'a P,
}

impl<'a, T, P: GenericPort<T>> Iterator<T> for RecvIterator<'a, P> {
    fn next(&mut self) -> Option<T> {
        self.port.try_recv()
    }
}

/// Ports that can `peek`
pub trait Peekable<T> {
    /// Returns true if a message is available
    fn peek(&self) -> bool;
}

/* priv is disabled to allow users to get at traits like Select. */
pub struct PortOne<T> { /* priv */ x: rtcomm::PortOne<T> }
pub struct ChanOne<T> { /* priv */ x: rtcomm::ChanOne<T> }

pub fn oneshot<T: Send>() -> (PortOne<T>, ChanOne<T>) {
    let (p, c) = rtcomm::oneshot();
    (PortOne { x: p }, ChanOne { x: c })
}

pub struct Port<T> { /* priv */ x: rtcomm::Port<T> }
pub struct Chan<T> { /* priv */ x: rtcomm::Chan<T> }

pub fn stream<T: Send>() -> (Port<T>, Chan<T>) {
    let (p, c) = rtcomm::stream();
    (Port { x: p }, Chan { x: c })
}

impl<T: Send> ChanOne<T> {
    pub fn send(self, val: T) {
        let ChanOne { x: c } = self;
        c.send(val)
    }

    pub fn try_send(self, val: T) -> bool {
        let ChanOne { x: c } = self;
        c.try_send(val)
    }

    pub fn send_deferred(self, val: T) {
        let ChanOne { x: c } = self;
        c.send_deferred(val)
    }

    pub fn try_send_deferred(self, val: T) -> bool {
        let ChanOne{ x: c } = self;
        c.try_send_deferred(val)
    }
}

impl<T: Send> PortOne<T> {
    pub fn recv(self) -> T {
        let PortOne { x: p } = self;
        p.recv()
    }

    pub fn try_recv(self) -> Option<T> {
        let PortOne { x: p } = self;
        p.try_recv()
    }
}

impl<T: Send> Peekable<T>  for PortOne<T> {
    fn peek(&self) -> bool {
        let &PortOne { x: ref p } = self;
        p.peek()
    }
}

impl<T: Send> GenericChan<T> for Chan<T> {
    fn send(&self, val: T) {
        let &Chan { x: ref c } = self;
        c.send(val)
    }
}

impl<T: Send> GenericSmartChan<T> for Chan<T> {
    fn try_send(&self, val: T) -> bool {
        let &Chan { x: ref c } = self;
        c.try_send(val)
    }
}

impl<T: Send> SendDeferred<T> for Chan<T> {
    fn send_deferred(&self, val: T) {
        let &Chan { x: ref c } = self;
        c.send_deferred(val)
    }

    fn try_send_deferred(&self, val: T) -> bool {
        let &Chan { x: ref c } = self;
        c.try_send_deferred(val)
    }
}

impl<T: Send> GenericPort<T> for Port<T> {
    fn recv(&self) -> T {
        let &Port { x: ref p } = self;
        p.recv()
    }

    fn try_recv(&self) -> Option<T> {
        let &Port { x: ref p } = self;
        p.try_recv()
    }
}

impl<T: Send> Peekable<T> for Port<T> {
    fn peek(&self) -> bool {
        let &Port { x: ref p } = self;
        p.peek()
    }
}


pub struct SharedChan<T> { /* priv */ x: rtcomm::SharedChan<T> }

impl<T: Send> SharedChan<T> {
    pub fn new(c: Chan<T>) -> SharedChan<T> {
        let Chan { x: c } = c;
        SharedChan { x: rtcomm::SharedChan::new(c) }
    }
}

impl<T: Send> GenericChan<T> for SharedChan<T> {
    fn send(&self, val: T) {
        let &SharedChan { x: ref c } = self;
        c.send(val)
    }
}

impl<T: Send> GenericSmartChan<T> for SharedChan<T> {
    fn try_send(&self, val: T) -> bool {
        let &SharedChan { x: ref c } = self;
        c.try_send(val)
    }
}

impl<T: Send> SendDeferred<T> for SharedChan<T> {
    fn send_deferred(&self, val: T) {
        let &SharedChan { x: ref c } = self;
        c.send_deferred(val)
    }

    fn try_send_deferred(&self, val: T) -> bool {
        let &SharedChan { x: ref c } = self;
        c.try_send_deferred(val)
    }
}

impl<T: Send> Clone for SharedChan<T> {
    fn clone(&self) -> SharedChan<T> {
        let &SharedChan { x: ref c } = self;
        SharedChan { x: c.clone() }
    }
}

pub struct SharedPort<T> { /* priv */ x: rtcomm::SharedPort<T> }

impl<T: Send> SharedPort<T> {
    pub fn new(p: Port<T>) -> SharedPort<T> {
        let Port { x: p } = p;
        SharedPort { x: rtcomm::SharedPort::new(p) }
    }
}

impl<T: Send> GenericPort<T> for SharedPort<T> {
    fn recv(&self) -> T {
        let &SharedPort { x: ref p } = self;
        p.recv()
    }

    fn try_recv(&self) -> Option<T> {
        let &SharedPort { x: ref p } = self;
        p.try_recv()
    }
}

impl<T: Send> Clone for SharedPort<T> {
    fn clone(&self) -> SharedPort<T> {
        let &SharedPort { x: ref p } = self;
        SharedPort { x: p.clone() }
    }
}

#[cfg(test)]
mod tests {
    use comm::*;
    use prelude::*;

    #[test]
    fn test_nested_recv_iter() {
        let (port, chan) = stream::<int>();
        let (total_port, total_chan) = oneshot::<int>();

        do spawn {
            let mut acc = 0;
            for x in port.recv_iter() {
                acc += x;
                for x in port.recv_iter() {
                    acc += x;
                    for x in port.try_recv().move_iter() {
                        acc += x;
                        total_chan.send(acc);
                    }
                }
            }
        }

        chan.send(3);
        chan.send(1);
        chan.send(2);
        assert_eq!(total_port.recv(), 6);
    }

    #[test]
    fn test_recv_iter_break() {
        let (port, chan) = stream::<int>();
        let (count_port, count_chan) = oneshot::<int>();

        do spawn {
            let mut count = 0;
            for x in port.recv_iter() {
                if count >= 3 {
                    count_chan.send(count);
                    break;
                } else {
                    count += x;
                }
            }
        }

        chan.send(2);
        chan.send(2);
        chan.send(2);
        chan.send(2);
        assert_eq!(count_port.recv(), 4);
    }
}
