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
use kinds::Send;
use option::Option;
pub use rt::comm::SendDeferred;
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

/// A trait for things that can receive multiple messages.
pub trait GenericPort<T> {
    /// Receives a message, or fails if the connection closes.
    fn recv(&self) -> T;

    /** Receives a message, or returns `none` if
    the connection is closed or closes.
    */
    fn try_recv(&self) -> Option<T>;
}

/// Ports that can `peek`
pub trait Peekable<T> {
    /// Returns true if a message is available
    fn peek(&self) -> bool;
}

pub struct PortOne<T> { x: rtcomm::PortOne<T> }
pub struct ChanOne<T> { x: rtcomm::ChanOne<T> }

pub fn oneshot<T: Send>() -> (PortOne<T>, ChanOne<T>) {
    let (p, c) = rtcomm::oneshot();
    (PortOne { x: p }, ChanOne { x: c })
}

pub struct Port<T> { x: rtcomm::Port<T> }
pub struct Chan<T> { x: rtcomm::Chan<T> }

pub fn stream<T: Send>() -> (Port<T>, Chan<T>) {
    let (p, c) = rtcomm::stream();
    (Port { x: p }, Chan { x: c })
}

pub struct SharedChan<T> { x: rtcomm::SharedChan<T> }

impl<T: Send> SharedChan<T> {
    pub fn new(c: Chan<T>) -> SharedChan<T> {
        let Chan { x: c } = c;
        SharedChan { x: rtcomm::SharedChan::new(c) }
    }
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

impl<T> Clone for SharedChan<T> {
    fn clone(&self) -> SharedChan<T> {
        let &SharedChan { x: ref c } = self;
        SharedChan { x: c.clone() }
    }
}
