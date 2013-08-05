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

use either::{Either, Left, Right};
use kinds::Send;
use option::{Option, Some};
use unstable::sync::Exclusive;
pub use rt::comm::SendDeferred;
use rtcomm = rt::comm;
use rt;

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

/// An endpoint that can send many messages.
pub struct Chan<T> {
    inner: Either<pipesy::Chan<T>, rtcomm::Chan<T>>
}

/// An endpoint that can receive many messages.
pub struct Port<T> {
    inner: Either<pipesy::Port<T>, rtcomm::Port<T>>
}

/** Creates a `(Port, Chan)` pair.

These allow sending or receiving an unlimited number of messages.

*/
pub fn stream<T:Send>() -> (Port<T>, Chan<T>) {
    let (port, chan) = match rt::context() {
        rt::OldTaskContext => match pipesy::stream() {
            (p, c) => (Left(p), Left(c))
        },
        _ => match rtcomm::stream() {
            (p, c) => (Right(p), Right(c))
        }
    };
    let port = Port { inner: port };
    let chan = Chan { inner: chan };
    return (port, chan);
}

impl<T: Send> GenericChan<T> for Chan<T> {
    fn send(&self, x: T) {
        match self.inner {
            Left(ref chan) => chan.send(x),
            Right(ref chan) => chan.send(x)
        }
    }
}

impl<T: Send> GenericSmartChan<T> for Chan<T> {
    fn try_send(&self, x: T) -> bool {
        match self.inner {
            Left(ref chan) => chan.try_send(x),
            Right(ref chan) => chan.try_send(x)
        }
    }
}

impl<T: Send> SendDeferred<T> for Chan<T> {
    fn send_deferred(&self, x: T) {
        match self.inner {
            Left(ref chan) => chan.send(x),
            Right(ref chan) => chan.send_deferred(x)
        }
    }
    fn try_send_deferred(&self, x: T) -> bool {
        match self.inner {
            Left(ref chan) => chan.try_send(x),
            Right(ref chan) => chan.try_send_deferred(x)
        }
    }
}

impl<T: Send> GenericPort<T> for Port<T> {
    fn recv(&self) -> T {
        match self.inner {
            Left(ref port) => port.recv(),
            Right(ref port) => port.recv()
        }
    }

    fn try_recv(&self) -> Option<T> {
        match self.inner {
            Left(ref port) => port.try_recv(),
            Right(ref port) => port.try_recv()
        }
    }
}

impl<T: Send> Peekable<T> for Port<T> {
    fn peek(&self) -> bool {
        match self.inner {
            Left(ref port) => port.peek(),
            Right(ref port) => port.peek()
        }
    }
}

/// A channel that can be shared between many senders.
pub struct SharedChan<T> {
    inner: Either<Exclusive<pipesy::Chan<T>>, rtcomm::SharedChan<T>>
}

impl<T: Send> SharedChan<T> {
    /// Converts a `chan` into a `shared_chan`.
    pub fn new(c: Chan<T>) -> SharedChan<T> {
        let Chan { inner } = c;
        let c = match inner {
            Left(c) => Left(Exclusive::new(c)),
            Right(c) => Right(rtcomm::SharedChan::new(c))
        };
        SharedChan { inner: c }
    }
}

impl<T: Send> GenericChan<T> for SharedChan<T> {
    fn send(&self, x: T) {
        match self.inner {
            Left(ref chan) => {
                unsafe {
                    let mut xx = Some(x);
                    do chan.with_imm |chan| {
                        chan.send(xx.take_unwrap())
                    }
                }
            }
            Right(ref chan) => chan.send(x)
        }
    }
}

impl<T: Send> GenericSmartChan<T> for SharedChan<T> {
    fn try_send(&self, x: T) -> bool {
        match self.inner {
            Left(ref chan) => {
                unsafe {
                    let mut xx = Some(x);
                    do chan.with_imm |chan| {
                        chan.try_send(xx.take_unwrap())
                    }
                }
            }
            Right(ref chan) => chan.try_send(x)
        }
    }
}

impl<T: Send> ::clone::Clone for SharedChan<T> {
    fn clone(&self) -> SharedChan<T> {
        SharedChan { inner: self.inner.clone() }
    }
}

pub struct PortOne<T> {
    inner: Either<pipesy::PortOne<T>, rtcomm::PortOne<T>>
}

pub struct ChanOne<T> {
    inner: Either<pipesy::ChanOne<T>, rtcomm::ChanOne<T>>
}

pub fn oneshot<T: Send>() -> (PortOne<T>, ChanOne<T>) {
    let (port, chan) = match rt::context() {
        rt::OldTaskContext => match pipesy::oneshot() {
            (p, c) => (Left(p), Left(c)),
        },
        _ => match rtcomm::oneshot() {
            (p, c) => (Right(p), Right(c))
        }
    };
    let port = PortOne { inner: port };
    let chan = ChanOne { inner: chan };
    return (port, chan);
}

impl<T: Send> PortOne<T> {
    pub fn recv(self) -> T {
        let PortOne { inner } = self;
        match inner {
            Left(p) => p.recv(),
            Right(p) => p.recv()
        }
    }

    pub fn try_recv(self) -> Option<T> {
        let PortOne { inner } = self;
        match inner {
            Left(p) => p.try_recv(),
            Right(p) => p.try_recv()
        }
    }
}

impl<T: Send> ChanOne<T> {
    pub fn send(self, data: T) {
        let ChanOne { inner } = self;
        match inner {
            Left(p) => p.send(data),
            Right(p) => p.send(data)
        }
    }

    pub fn try_send(self, data: T) -> bool {
        let ChanOne { inner } = self;
        match inner {
            Left(p) => p.try_send(data),
            Right(p) => p.try_send(data)
        }
    }
    pub fn send_deferred(self, data: T) {
        let ChanOne { inner } = self;
        match inner {
            Left(p) => p.send(data),
            Right(p) => p.send_deferred(data)
        }
    }
    pub fn try_send_deferred(self, data: T) -> bool {
        let ChanOne { inner } = self;
        match inner {
            Left(p) => p.try_send(data),
            Right(p) => p.try_send_deferred(data)
        }
    }
}

pub fn recv_one<T: Send>(port: PortOne<T>) -> T {
    let PortOne { inner } = port;
    match inner {
        Left(p) => pipesy::recv_one(p),
        Right(p) => p.recv()
    }
}

pub fn try_recv_one<T: Send>(port: PortOne<T>) -> Option<T> {
    let PortOne { inner } = port;
    match inner {
        Left(p) => pipesy::try_recv_one(p),
        Right(p) => p.try_recv()
    }
}

pub fn send_one<T: Send>(chan: ChanOne<T>, data: T) {
    let ChanOne { inner } = chan;
    match inner {
        Left(c) => pipesy::send_one(c, data),
        Right(c) => c.send(data)
    }
}

pub fn try_send_one<T: Send>(chan: ChanOne<T>, data: T) -> bool {
    let ChanOne { inner } = chan;
    match inner {
        Left(c) => pipesy::try_send_one(c, data),
        Right(c) => c.try_send(data)
    }
}

mod pipesy {

    use kinds::Send;
    use option::{Option, Some, None};
    use pipes::{recv, try_recv, peek};
    use super::{GenericChan, GenericSmartChan, GenericPort, Peekable};
    use cast::transmute_mut;

    /*proto! oneshot (
        Oneshot:send<T:Send> {
            send(T) -> !
        }
    )*/

    #[allow(non_camel_case_types)]
    pub mod oneshot {
        priv use std::kinds::Send;
        use ptr::to_mut_unsafe_ptr;

        pub fn init<T: Send>() -> (server::Oneshot<T>, client::Oneshot<T>) {
            pub use std::pipes::HasBuffer;

            let buffer = ~::std::pipes::Buffer {
                header: ::std::pipes::BufferHeader(),
                data: __Buffer {
                    Oneshot: ::std::pipes::mk_packet::<Oneshot<T>>()
                },
            };
            do ::std::pipes::entangle_buffer(buffer) |buffer, data| {
                data.Oneshot.set_buffer(buffer);
                to_mut_unsafe_ptr(&mut data.Oneshot)
            }
        }
        #[allow(non_camel_case_types)]
        pub enum Oneshot<T> { pub send(T), }
        #[allow(non_camel_case_types)]
        pub struct __Buffer<T> {
            Oneshot: ::std::pipes::Packet<Oneshot<T>>,
        }

        #[allow(non_camel_case_types)]
        pub mod client {

            priv use std::kinds::Send;

            #[allow(non_camel_case_types)]
            pub fn try_send<T: Send>(pipe: Oneshot<T>, x_0: T) ->
                ::std::option::Option<()> {
                {
                    use super::send;
                    let message = send(x_0);
                    if ::std::pipes::send(pipe, message) {
                        ::std::pipes::rt::make_some(())
                    } else { ::std::pipes::rt::make_none() }
                }
            }

            #[allow(non_camel_case_types)]
            pub fn send<T: Send>(pipe: Oneshot<T>, x_0: T) {
                {
                    use super::send;
                    let message = send(x_0);
                    ::std::pipes::send(pipe, message);
                }
            }

            #[allow(non_camel_case_types)]
            pub type Oneshot<T> =
                ::std::pipes::SendPacketBuffered<super::Oneshot<T>,
            super::__Buffer<T>>;
        }

        #[allow(non_camel_case_types)]
        pub mod server {
            #[allow(non_camel_case_types)]
            pub type Oneshot<T> =
                ::std::pipes::RecvPacketBuffered<super::Oneshot<T>,
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
    pub fn oneshot<T: Send>() -> (PortOne<T>, ChanOne<T>) {
        let (port, chan) = oneshot::init();
        (PortOne::new(port), ChanOne::new(chan))
    }

    impl<T: Send> PortOne<T> {
        pub fn recv(self) -> T { recv_one(self) }
        pub fn try_recv(self) -> Option<T> { try_recv_one(self) }
        pub fn unwrap(self) -> oneshot::server::Oneshot<T> {
            match self {
                PortOne { contents: s } => s
            }
        }
    }

    impl<T: Send> ChanOne<T> {
        pub fn send(self, data: T) { send_one(self, data) }
        pub fn try_send(self, data: T) -> bool { try_send_one(self, data) }
        pub fn unwrap(self) -> oneshot::client::Oneshot<T> {
            match self {
                ChanOne { contents: s } => s
            }
        }
    }

    /**
    * Receive a message from a oneshot pipe, failing if the connection was
    * closed.
    */
    pub fn recv_one<T: Send>(port: PortOne<T>) -> T {
        match port {
            PortOne { contents: port } => {
                let oneshot::send(message) = recv(port);
                message
            }
        }
    }

    /// Receive a message from a oneshot pipe unless the connection was closed.
    pub fn try_recv_one<T: Send> (port: PortOne<T>) -> Option<T> {
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
    pub fn send_one<T: Send>(chan: ChanOne<T>, data: T) {
        match chan {
            ChanOne { contents: chan } => oneshot::client::send(chan, data),
        }
    }

    /**
    * Send a message on a oneshot pipe, or return false if the connection was
    * closed.
    */
    pub fn try_send_one<T: Send>(chan: ChanOne<T>, data: T) -> bool {
        match chan {
            ChanOne { contents: chan } => {
                oneshot::client::try_send(chan, data).is_some()
            }
        }
    }

    // Streams - Make pipes a little easier in general.

    /*proto! streamp (
        Open:send<T: Send> {
            data(T) -> Open<T>
        }
    )*/

    #[allow(non_camel_case_types)]
    pub mod streamp {
        priv use std::kinds::Send;

        pub fn init<T: Send>() -> (server::Open<T>, client::Open<T>) {
            pub use std::pipes::HasBuffer;
            ::std::pipes::entangle()
        }

        #[allow(non_camel_case_types)]
        pub enum Open<T> { pub data(T, server::Open<T>), }

        #[allow(non_camel_case_types)]
        pub mod client {
            priv use std::kinds::Send;

            #[allow(non_camel_case_types)]
            pub fn try_data<T: Send>(pipe: Open<T>, x_0: T) ->
                ::std::option::Option<Open<T>> {
                {
                    use super::data;
                    let (s, c) = ::std::pipes::entangle();
                    let message = data(x_0, s);
                    if ::std::pipes::send(pipe, message) {
                        ::std::pipes::rt::make_some(c)
                    } else { ::std::pipes::rt::make_none() }
                }
            }

            #[allow(non_camel_case_types)]
            pub fn data<T: Send>(pipe: Open<T>, x_0: T) -> Open<T> {
                {
                    use super::data;
                    let (s, c) = ::std::pipes::entangle();
                    let message = data(x_0, s);
                    ::std::pipes::send(pipe, message);
                    c
                }
            }

            #[allow(non_camel_case_types)]
            pub type Open<T> = ::std::pipes::SendPacket<super::Open<T>>;
        }

        #[allow(non_camel_case_types)]
        pub mod server {
            #[allow(non_camel_case_types)]
            pub type Open<T> = ::std::pipes::RecvPacket<super::Open<T>>;
        }
    }

    /// An endpoint that can send many messages.
    #[unsafe_mut_field(endp)]
    pub struct Chan<T> {
        endp: Option<streamp::client::Open<T>>
    }

    /// An endpoint that can receive many messages.
    #[unsafe_mut_field(endp)]
    pub struct Port<T> {
        endp: Option<streamp::server::Open<T>>,
    }

    /** Creates a `(Port, Chan)` pair.

    These allow sending or receiving an unlimited number of messages.

    */
    pub fn stream<T:Send>() -> (Port<T>, Chan<T>) {
        let (s, c) = streamp::init();

        (Port {
            endp: Some(s)
        }, Chan {
            endp: Some(c)
        })
    }

    impl<T: Send> GenericChan<T> for Chan<T> {
        #[inline]
        fn send(&self, x: T) {
            unsafe {
                let self_endp = transmute_mut(&self.endp);
                *self_endp = Some(streamp::client::data(self_endp.take_unwrap(), x))
            }
        }
    }

    impl<T: Send> GenericSmartChan<T> for Chan<T> {
        #[inline]
        fn try_send(&self, x: T) -> bool {
            unsafe {
                let self_endp = transmute_mut(&self.endp);
                match streamp::client::try_data(self_endp.take_unwrap(), x) {
                    Some(next) => {
                        *self_endp = Some(next);
                        true
                    }
                    None => false
                }
            }
        }
    }

    impl<T: Send> GenericPort<T> for Port<T> {
        #[inline]
        fn recv(&self) -> T {
            unsafe {
                let self_endp = transmute_mut(&self.endp);
                let endp = self_endp.take();
                let streamp::data(x, endp) = recv(endp.unwrap());
                *self_endp = Some(endp);
                x
            }
        }

        #[inline]
        fn try_recv(&self) -> Option<T> {
            unsafe {
                let self_endp = transmute_mut(&self.endp);
                let endp = self_endp.take();
                match try_recv(endp.unwrap()) {
                    Some(streamp::data(x, endp)) => {
                        *self_endp = Some(endp);
                        Some(x)
                    }
                    None => None
                }
            }
        }
    }

    impl<T: Send> Peekable<T> for Port<T> {
        #[inline]
        fn peek(&self) -> bool {
            unsafe {
                let self_endp = transmute_mut(&self.endp);
                let mut endp = self_endp.take();
                let peek = match endp {
                    Some(ref mut endp) => peek(endp),
                    None => fail!("peeking empty stream")
                };
                *self_endp = endp;
                peek
            }
        }
    }

}

#[cfg(test)]
mod test {
    use either::Right;
    use super::{Chan, Port, oneshot, stream};

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
