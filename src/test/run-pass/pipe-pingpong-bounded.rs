// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ping-pong is a bounded protocol. This is place where I can
// experiment with what code the compiler should generate for bounded
// protocols.

use core::cell::Cell;

// This was generated initially by the pipe compiler, but it's been
// modified in hopefully straightforward ways.

mod pingpong {
    use core::pipes;
    use core::pipes::*;
    use core::ptr;

    pub struct Packets {
        ping: Packet<ping>,
        pong: Packet<pong>,
    }

    pub fn init() -> (client::ping, server::ping) {
        let buffer = ~Buffer {
            header: BufferHeader(),
            data: Packets {
                ping: mk_packet::<ping>(),
                pong: mk_packet::<pong>()
            }
        };
        do pipes::entangle_buffer(buffer) |buffer, data| {
            data.ping.set_buffer(buffer);
            data.pong.set_buffer(buffer);
            ptr::to_mut_unsafe_ptr(&mut (data.ping))
        }
    }
    pub struct ping(server::pong);
    pub struct pong(client::ping);
    pub mod client {
        use core::pipes;
        use core::pipes::*;
        use core::ptr;

        pub fn ping(mut pipe: ping) -> pong {
            {
                let mut b = pipe.reuse_buffer();
                let s = SendPacketBuffered(&mut b.buffer.data.pong);
                let c = RecvPacketBuffered(&mut b.buffer.data.pong);
                let message = ::pingpong::ping(s);
                send(pipe, message);
                c
            }
        }
        pub type ping = pipes::SendPacketBuffered<::pingpong::ping,
                                                  ::pingpong::Packets>;
        pub type pong = pipes::RecvPacketBuffered<::pingpong::pong,
                                                  ::pingpong::Packets>;
    }
    pub mod server {
        use core::pipes;
        use core::pipes::*;
        use core::ptr;

        pub type ping = pipes::RecvPacketBuffered<::pingpong::ping,
        ::pingpong::Packets>;
        pub fn pong(mut pipe: pong) -> ping {
            {
                let mut b = pipe.reuse_buffer();
                let s = SendPacketBuffered(&mut b.buffer.data.ping);
                let c = RecvPacketBuffered(&mut b.buffer.data.ping);
                let message = ::pingpong::pong(s);
                send(pipe, message);
                c
            }
        }
        pub type pong = pipes::SendPacketBuffered<::pingpong::pong,
                                                  ::pingpong::Packets>;
    }
}

mod test {
    use core::pipes::recv;
    use pingpong::{ping, pong};

    pub fn client(chan: ::pingpong::client::ping) {
        use pingpong::client;

        let chan = client::ping(chan); return;
        error!("Sent ping");
        let pong(_chan) = recv(chan);
        error!("Received pong");
    }

    pub fn server(chan: ::pingpong::server::ping) {
        use pingpong::server;

        let ping(chan) = recv(chan); return;
        error!("Received ping");
        let _chan = server::pong(chan);
        error!("Sent pong");
    }
}

pub fn main() {
    let (client_, server_) = ::pingpong::init();
    let client_ = Cell(client_);
    let server_ = Cell(server_);
    do task::spawn {
        let client__ = client_.take();
        test::client(client__);
    };
    do task::spawn {
        let server__ = server_.take();
        test::server(server__);
    };
}
