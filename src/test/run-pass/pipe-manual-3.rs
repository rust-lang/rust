/*
The first test case using pipes. The idea is to break this into
several stages for prototyping. Here's the plan:

1. Write an already-compiled protocol using existing ports and chans.

2. Take the already-compiled version and add the low-level
synchronization code instead.

3. Write a syntax extension to compile the protocols.

At some point, we'll need to add support for select.

This file does horrible things to pretend we have self-move.

*/

mod pingpong {
    enum ping { ping, }
    enum ping_message = *pipes::packet<pong_message>;
    enum pong { pong, }
    enum pong_message = *pipes::packet<ping_message>;

    fn init() -> (client::ping, server::ping) {
        pipes::entangle()
    }

    mod client {
        type ping = pipes::send_packet<pingpong::ping_message>;
        type pong = pipes::recv_packet<pingpong::pong_message>;
    }

    impl abominable for client::ping {
        fn send() -> fn@(-client::ping, ping) -> client::pong {
            |pipe, data| {
                let p = pipes::packet();
                pipes::send(pipe, pingpong::ping_message(p));
                pipes::recv_packet(p)
            }
        }
    }

    impl abominable for client::pong {
        fn recv() -> fn@(-client::pong) -> (client::ping, pong) {
            |pipe| {
                let packet = pipes::recv(pipe);
                if packet == none {
                    fail "sender closed the connection"
                }
                let p : pong_message = option::unwrap(packet);
                (pipes::send_packet(*p), pong)
            }
        }
    }

    mod server {
        type ping = pipes::recv_packet<pingpong::ping_message>;
        type pong = pipes::send_packet<pingpong::pong_message>;
    }

    impl abominable for server::ping {
        fn recv() -> fn@(-server::ping) -> (server::pong, ping) {
            |pipe| {
                let packet = pipes::recv(pipe);
                if packet == none {
                    fail "sender closed the connection"
                }
                let p : ping_message = option::unwrap(packet);
                (pipes::send_packet(*p), ping)
            }
        }
    }

    impl abominable for server::pong {
        fn send() -> fn@(-server::pong, pong) -> server::ping {
            |pipe, data| {
                let p = pipes::packet();
                pipes::send(pipe, pingpong::pong_message(p));
                pipes::recv_packet(p)
            }
        }
    }
}

mod test {
    import pingpong::{ping, pong, abominable};

    fn macros() {
        #macro[
            [#send[chan, data],
             chan.send()(chan, data)]
        ];
        #macro[
            [#recv[chan],
             chan.recv()(chan)]
        ];
    }

    fn client(-chan: pingpong::client::ping) {
        let chan = #send(chan, ping);
        log(error, "Sent ping");
        let (chan, _data) = #recv(chan);
        log(error, "Received pong");
    }
    
    fn server(-chan: pingpong::server::ping) {
        let (chan, _data) = #recv(chan);
        log(error, "Received ping");
        let chan = #send(chan, pong);
        log(error, "Sent pong");
    }
}

fn main() {
    let (client_, server_) = pingpong::init();
    let client_ = ~mut some(client_);
    let server_ = ~mut some(server_);

    do task::spawn |move client_| {
        let mut client__ = none;
        *client_ <-> client__;
        test::client(option::unwrap(client__));
    };
    do task::spawn |move server_| {
        let mut server_ˊ = none;
        *server_ <-> server_ˊ;
        test::server(option::unwrap(server_ˊ));
    };
}
