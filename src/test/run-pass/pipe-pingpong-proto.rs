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

// An example to make sure the protocol parsing syntax extension works.

use core::option;

proto! pingpong (
    ping:send {
        ping -> pong
    }

    pong:recv {
        pong -> ping
    }
)

mod test {
    use core::pipes::recv;
    use pingpong::{ping, pong};

    pub fn client(-chan: ::pingpong::client::ping) {
        use pingpong::client;

        let chan = client::ping(chan);
        log(error, ~"Sent ping");
        let pong(_chan) = recv(chan);
        log(error, ~"Received pong");
    }
    
    pub fn server(-chan: ::pingpong::server::ping) {
        use pingpong::server;

        let ping(chan) = recv(chan);
        log(error, ~"Received ping");
        let _chan = server::pong(chan);
        log(error, ~"Sent pong");
    }
}

pub fn main() {
    let (client_, server_) = pingpong::init();
    let client_ = ~mut Some(client_);
    let server_ = ~mut Some(server_);

    do task::spawn || {
        let mut client__ = None;
        *client_ <-> client__;
        test::client(option::unwrap(client__));
    };
    do task::spawn || {
        let mut server_ˊ = None;
        *server_ <-> server_ˊ;
        test::server(option::unwrap(server_ˊ));
    };
}
