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


// This was generated initially by the pipe compiler, but it's been
// modified in hopefully straightforward ways.
mod pingpong {
    use core::pipes::*;
    use core::ptr;

    pub type packets = {
        ping: Packet<ping>,
        pong: Packet<pong>,
    };

    pub fn init() -> (client::ping, server::ping) {
        let buffer = ~{
            header: BufferHeader(),
            data: {
                ping: mk_packet::<ping>(),
                pong: mk_packet::<pong>()
            }
        };
        do pipes::entangle_buffer(move buffer) |buffer, data| {
            data.ping.set_buffer_(buffer);
            data.pong.set_buffer_(buffer);
            ptr::addr_of(&(data.ping))
        }
    }
    pub enum ping = server::pong;
    pub enum pong = client::ping;
    pub mod client {
        use core::pipes::*;
        use core::ptr;

        pub fn ping(+pipe: ping) -> pong {
            {
                let b = pipe.reuse_buffer();
                let s = SendPacketBuffered(ptr::addr_of(&(b.buffer.data.pong)));
                let c = RecvPacketBuffered(ptr::addr_of(&(b.buffer.data.pong)));
                let message = ::pingpong::ping(move s);
                ::pipes::send(move pipe, move message);
                move c
            }
        }
        pub type ping = pipes::SendPacketBuffered<::pingpong::ping,
                                                  ::pingpong::packets>;
        pub type pong = pipes::RecvPacketBuffered<::pingpong::pong,
                                                  ::pingpong::packets>;
    }
    pub mod server {
        use core::pipes::*;
        use core::ptr;

        pub type ping = pipes::RecvPacketBuffered<::pingpong::ping,
        ::pingpong::packets>;
        pub fn pong(+pipe: pong) -> ping {
            {
                let b = pipe.reuse_buffer();
                let s = SendPacketBuffered(ptr::addr_of(&(b.buffer.data.ping)));
                let c = RecvPacketBuffered(ptr::addr_of(&(b.buffer.data.ping)));
                let message = ::pingpong::pong(move s);
                ::pipes::send(move pipe, move message);
                move c
            }
        }
        pub type pong = pipes::SendPacketBuffered<::pingpong::pong,
                                                  ::pingpong::packets>;
    }
}

mod test {
    use pipes::recv;
    use pingpong::{ping, pong};

    pub fn client(-chan: ::pingpong::client::ping) {
        use pingpong::client;

        let chan = client::ping(move chan); return;
        log(error, "Sent ping");
        let pong(_chan) = recv(move chan);
        log(error, "Received pong");
    }
    
    pub fn server(-chan: ::pingpong::server::ping) {
        use pingpong::server;

        let ping(chan) = recv(move chan); return;
        log(error, "Received ping");
        let _chan = server::pong(move chan);
        log(error, "Sent pong");
    }
}

fn main() {
    let (client_, server_) = ::pingpong::init();
    let client_ = ~mut Some(move client_);
    let server_ = ~mut Some(move server_);
    do task::spawn |move client_| {
        let mut client__ = None;
        *client_ <-> client__;
        test::client(option::unwrap(move client__));
    };
    do task::spawn |move server_| {
        let mut server_ˊ = None;
        *server_ <-> server_ˊ;
        test::server(option::unwrap(move server_ˊ));
    };
}
