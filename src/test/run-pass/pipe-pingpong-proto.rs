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


proto! pingpong (
    ping:send {
        ping -> pong
    }

    pong:recv {
        pong -> ping
    }
)

mod test {
    #[legacy_exports];
    use pipes::recv;
    use pingpong::{ping, pong};

    fn client(-chan: pingpong::client::ping) {
        use pingpong::client;

        let chan = client::ping(move chan);
        log(error, ~"Sent ping");
        let pong(_chan) = recv(move chan);
        log(error, ~"Received pong");
    }
    
    fn server(-chan: pingpong::server::ping) {
        use pingpong::server;

        let ping(chan) = recv(move chan);
        log(error, ~"Received ping");
        let _chan = server::pong(move chan);
        log(error, ~"Sent pong");
    }
}

fn main() {
    let (client_, server_) = pingpong::init();
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
