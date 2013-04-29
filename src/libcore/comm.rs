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

use cast;
use either::{Either, Left, Right};
use kinds::Owned;
use option::{Option, Some, None};
use uint;
use unstable;
use vec;
use unstable::Exclusive;

use pipes::{recv, try_recv, wait_many, peek, PacketHeader};

// FIXME #5160: Making this public exposes some plumbing from
// pipes. Needs some refactoring
pub use pipes::Selectable;

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


// Streams - Make pipes a little easier in general.

/*proto! streamp (
    Open:send<T: Owned> {
        data(T) -> Open<T>
    }
)*/

#[allow(non_camel_case_types)]
pub mod streamp {
    priv use core::kinds::Owned;

    pub fn init<T: Owned>() -> (client::Open<T>, server::Open<T>) {
        pub use core::pipes::HasBuffer;
        ::core::pipes::entangle()
    }

    #[allow(non_camel_case_types)]
    pub enum Open<T> { pub data(T, server::Open<T>), }

    #[allow(non_camel_case_types)]
    pub mod client {
        priv use core::kinds::Owned;

        #[allow(non_camel_case_types)]
        pub fn try_data<T: Owned>(pipe: Open<T>, x_0: T) ->
            ::core::option::Option<Open<T>> {
            {
                use super::data;
                let (c, s) = ::core::pipes::entangle();
                let message = data(x_0, s);
                if ::core::pipes::send(pipe, message) {
                    ::core::pipes::rt::make_some(c)
                } else { ::core::pipes::rt::make_none() }
            }
        }

        #[allow(non_camel_case_types)]
        pub fn data<T: Owned>(pipe: Open<T>, x_0: T) -> Open<T> {
            {
                use super::data;
                let (c, s) = ::core::pipes::entangle();
                let message = data(x_0, s);
                ::core::pipes::send(pipe, message);
                c
            }
        }

        #[allow(non_camel_case_types)]
        pub type Open<T> = ::core::pipes::SendPacket<super::Open<T>>;
    }

    #[allow(non_camel_case_types)]
    pub mod server {
        #[allow(non_camel_case_types)]
        pub type Open<T> = ::core::pipes::RecvPacket<super::Open<T>>;
    }
}

/// An endpoint that can send many messages.
pub struct Chan<T> {
    mut endp: Option<streamp::client::Open<T>>
}

/// An endpoint that can receive many messages.
pub struct Port<T> {
    mut endp: Option<streamp::server::Open<T>>,
}

/** Creates a `(Port, Chan)` pair.

These allow sending or receiving an unlimited number of messages.

*/
pub fn stream<T:Owned>() -> (Port<T>, Chan<T>) {
    let (c, s) = streamp::init();

    (Port { endp: Some(s) }, Chan { endp: Some(c) })
}

impl<T: Owned> GenericChan<T> for Chan<T> {
    #[inline(always)]
    fn send(&self, x: T) {
        let mut endp = None;
        endp <-> self.endp;
        self.endp = Some(
            streamp::client::data(endp.unwrap(), x))
    }
}

impl<T: Owned> GenericSmartChan<T> for Chan<T> {
    #[inline(always)]
    fn try_send(&self, x: T) -> bool {
        let mut endp = None;
        endp <-> self.endp;
        match streamp::client::try_data(endp.unwrap(), x) {
            Some(next) => {
                self.endp = Some(next);
                true
            }
            None => false
        }
    }
}

impl<T: Owned> GenericPort<T> for Port<T> {
    #[inline(always)]
    fn recv(&self) -> T {
        let mut endp = None;
        endp <-> self.endp;
        let streamp::data(x, endp) = recv(endp.unwrap());
        self.endp = Some(endp);
        x
    }

    #[inline(always)]
    fn try_recv(&self) -> Option<T> {
        let mut endp = None;
        endp <-> self.endp;
        match try_recv(endp.unwrap()) {
            Some(streamp::data(x, endp)) => {
                self.endp = Some(endp);
                Some(x)
            }
            None => None
        }
    }
}

impl<T: Owned> Peekable<T> for Port<T> {
    #[inline(always)]
    fn peek(&self) -> bool {
        let mut endp = None;
        endp <-> self.endp;
        let peek = match &endp {
            &Some(ref endp) => peek(endp),
            &None => fail!(~"peeking empty stream")
        };
        self.endp <-> endp;
        peek
    }
}

impl<T: Owned> Selectable for Port<T> {
    fn header(&self) -> *PacketHeader {
        unsafe {
            match self.endp {
              Some(ref endp) => endp.header(),
              None => fail!(~"peeking empty stream")
            }
        }
    }
}

/// Treat many ports as one.
pub struct PortSet<T> {
    mut ports: ~[Port<T>],
}

pub impl<T: Owned> PortSet<T> {

    fn new() -> PortSet<T> {
        PortSet {
            ports: ~[]
        }
    }

    fn add(&self, port: Port<T>) {
        self.ports.push(port)
    }

    fn chan(&self) -> Chan<T> {
        let (po, ch) = stream();
        self.add(po);
        ch
    }
}

impl<T:Owned> GenericPort<T> for PortSet<T> {
    fn try_recv(&self) -> Option<T> {
        let mut result = None;
        // we have to swap the ports array so we aren't borrowing
        // aliasable mutable memory.
        let mut ports = ~[];
        ports <-> self.ports;
        while result.is_none() && ports.len() > 0 {
            let i = wait_many(ports);
            match ports[i].try_recv() {
                Some(m) => {
                    result = Some(m);
                }
                None => {
                    // Remove this port.
                    let _ = ports.swap_remove(i);
                }
            }
        }
        ports <-> self.ports;
        result
    }
    fn recv(&self) -> T {
        self.try_recv().expect("port_set: endpoints closed")
    }
}

impl<T: Owned> Peekable<T> for PortSet<T> {
    fn peek(&self) -> bool {
        // It'd be nice to use self.port.each, but that version isn't
        // pure.
        for uint::range(0, vec::uniq_len(&const self.ports)) |i| {
            // XXX: Botch pending demuting.
            unsafe {
                let port: &Port<T> = cast::transmute(&mut self.ports[i]);
                if port.peek() { return true }
            }
        }
        false
    }
}

/// A channel that can be shared between many senders.
pub struct SharedChan<T> {
    ch: Exclusive<Chan<T>>
}

impl<T: Owned> SharedChan<T> {
    /// Converts a `chan` into a `shared_chan`.
    pub fn new(c: Chan<T>) -> SharedChan<T> {
        SharedChan { ch: unstable::exclusive(c) }
    }
}

impl<T: Owned> GenericChan<T> for SharedChan<T> {
    fn send(&self, x: T) {
        let mut xx = Some(x);
        do self.ch.with_imm |chan| {
            let mut x = None;
            x <-> xx;
            chan.send(x.unwrap())
        }
    }
}

impl<T: Owned> GenericSmartChan<T> for SharedChan<T> {
    fn try_send(&self, x: T) -> bool {
        let mut xx = Some(x);
        do self.ch.with_imm |chan| {
            let mut x = None;
            x <-> xx;
            chan.try_send(x.unwrap())
        }
    }
}

impl<T: Owned> ::clone::Clone for SharedChan<T> {
    fn clone(&self) -> SharedChan<T> {
        SharedChan { ch: self.ch.clone() }
    }
}

/*proto! oneshot (
    Oneshot:send<T:Owned> {
        send(T) -> !
    }
)*/

#[allow(non_camel_case_types)]
pub mod oneshot {
    priv use core::kinds::Owned;
    use ptr::to_unsafe_ptr;

    pub fn init<T: Owned>() -> (client::Oneshot<T>, server::Oneshot<T>) {
        pub use core::pipes::HasBuffer;

        let buffer =
            ~::core::pipes::Buffer{
            header: ::core::pipes::BufferHeader(),
            data: __Buffer{
                Oneshot: ::core::pipes::mk_packet::<Oneshot<T>>()
            },
        };
        do ::core::pipes::entangle_buffer(buffer) |buffer, data| {
            {
                data.Oneshot.set_buffer(buffer);
                to_unsafe_ptr(&data.Oneshot)
            }
        }
    }
    #[allow(non_camel_case_types)]
    pub enum Oneshot<T> { pub send(T), }
    #[allow(non_camel_case_types)]
    pub struct __Buffer<T> {
        Oneshot: ::core::pipes::Packet<Oneshot<T>>,
    }

    #[allow(non_camel_case_types)]
    pub mod client {

        priv use core::kinds::Owned;

        #[allow(non_camel_case_types)]
        pub fn try_send<T: Owned>(pipe: Oneshot<T>, x_0: T) ->
            ::core::option::Option<()> {
            {
                use super::send;
                let message = send(x_0);
                if ::core::pipes::send(pipe, message) {
                    ::core::pipes::rt::make_some(())
                } else { ::core::pipes::rt::make_none() }
            }
        }

        #[allow(non_camel_case_types)]
        pub fn send<T: Owned>(pipe: Oneshot<T>, x_0: T) {
            {
                use super::send;
                let message = send(x_0);
                ::core::pipes::send(pipe, message);
            }
        }

        #[allow(non_camel_case_types)]
        pub type Oneshot<T> =
            ::core::pipes::SendPacketBuffered<super::Oneshot<T>,
                                              super::__Buffer<T>>;
    }

    #[allow(non_camel_case_types)]
    pub mod server {
        #[allow(non_camel_case_types)]
        pub type Oneshot<T> =
            ::core::pipes::RecvPacketBuffered<super::Oneshot<T>,
                                              super::__Buffer<T>>;
    }
}

/// The send end of a oneshot pipe.
pub struct ChanOne<T> {
    contents: oneshot::client::Oneshot<T>
}

impl<T> ChanOne<T> {
    pub fn new(contents: oneshot::client::Oneshot<T>) -> ChanOne<T> {
        ChanOne {
            contents: contents
        }
    }
}

/// The receive end of a oneshot pipe.
pub struct PortOne<T> {
    contents: oneshot::server::Oneshot<T>
}

impl<T> PortOne<T> {
    pub fn new(contents: oneshot::server::Oneshot<T>) -> PortOne<T> {
        PortOne {
            contents: contents
        }
    }
}

/// Initialiase a (send-endpoint, recv-endpoint) oneshot pipe pair.
pub fn oneshot<T: Owned>() -> (PortOne<T>, ChanOne<T>) {
    let (chan, port) = oneshot::init();
    (PortOne::new(port), ChanOne::new(chan))
}

pub impl<T: Owned> PortOne<T> {
    fn recv(self) -> T { recv_one(self) }
    fn try_recv(self) -> Option<T> { try_recv_one(self) }
    fn unwrap(self) -> oneshot::server::Oneshot<T> {
        match self {
            PortOne { contents: s } => s
        }
    }
}

pub impl<T: Owned> ChanOne<T> {
    fn send(self, data: T) { send_one(self, data) }
    fn try_send(self, data: T) -> bool { try_send_one(self, data) }
    fn unwrap(self) -> oneshot::client::Oneshot<T> {
        match self {
            ChanOne { contents: s } => s
        }
    }
}

/**
 * Receive a message from a oneshot pipe, failing if the connection was
 * closed.
 */
pub fn recv_one<T: Owned>(port: PortOne<T>) -> T {
    match port {
        PortOne { contents: port } => {
            let oneshot::send(message) = recv(port);
            message
        }
    }
}

/// Receive a message from a oneshot pipe unless the connection was closed.
pub fn try_recv_one<T: Owned> (port: PortOne<T>) -> Option<T> {
    match port {
        PortOne { contents: port } => {
            let message = try_recv(port);

            if message.is_none() {
                None
            } else {
                let oneshot::send(message) = message.unwrap();
                Some(message)
            }
        }
    }
}

/// Send a message on a oneshot pipe, failing if the connection was closed.
pub fn send_one<T: Owned>(chan: ChanOne<T>, data: T) {
    match chan {
        ChanOne { contents: chan } => oneshot::client::send(chan, data),
    }
}

/**
 * Send a message on a oneshot pipe, or return false if the connection was
 * closed.
 */
pub fn try_send_one<T: Owned>(chan: ChanOne<T>, data: T) -> bool {
    match chan {
        ChanOne { contents: chan } => {
            oneshot::client::try_send(chan, data).is_some()
        }
    }
}



/// Returns the index of an endpoint that is ready to receive.
pub fn selecti<T: Selectable>(endpoints: &[T]) -> uint {
    wait_many(endpoints)
}

/// Returns 0 or 1 depending on which endpoint is ready to receive
pub fn select2i<A: Selectable, B: Selectable>(a: &A, b: &B) ->
        Either<(), ()> {
    match wait_many([a.header(), b.header()]) {
      0 => Left(()),
      1 => Right(()),
      _ => fail!(~"wait returned unexpected index")
    }
}

/// Receive a message from one of two endpoints.
pub trait Select2<T: Owned, U: Owned> {
    /// Receive a message or return `None` if a connection closes.
    fn try_select(&self) -> Either<Option<T>, Option<U>>;
    /// Receive a message or fail if a connection closes.
    fn select(&self) -> Either<T, U>;
}

impl<T: Owned, U: Owned,
     Left: Selectable + GenericPort<T>,
     Right: Selectable + GenericPort<U>>
    Select2<T, U> for (Left, Right) {

    fn select(&self) -> Either<T, U> {
        match *self {
          (ref lp, ref rp) => match select2i(lp, rp) {
            Left(()) => Left (lp.recv()),
            Right(()) => Right(rp.recv())
          }
        }
    }

    fn try_select(&self) -> Either<Option<T>, Option<U>> {
        match *self {
          (ref lp, ref rp) => match select2i(lp, rp) {
            Left(()) => Left (lp.try_recv()),
            Right(()) => Right(rp.try_recv())
          }
        }
    }
}

#[cfg(test)]
mod test {
    use either::Right;
    use super::{Chan, Port, oneshot, recv_one, stream};

    #[test]
    fn test_select2() {
        let (p1, c1) = stream();
        let (p2, c2) = stream();

        c1.send(~"abc");

        match (p1, p2).select() {
          Right(_) => fail!(),
          _ => ()
        }

        c2.send(123);
    }

    #[test]
    fn test_oneshot() {
        let (p, c) = oneshot();

        c.send(());

        p.recv()
    }

    #[test]
    fn test_peek_terminated() {
        let (port, chan): (Port<int>, Chan<int>) = stream();

        {
            // Destroy the channel
            let _chan = chan;
        }

        assert!(!port.peek());
    }
}
