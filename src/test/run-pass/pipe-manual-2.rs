/*
The first test case using pipes. The idea is to break this into
several stages for prototyping. Here's the plan:

1. Write an already-compiled protocol using existing ports and chans.

2. Take the already-compiled version and add the low-level
synchronization code instead. (That's what this file attempts to do)

3. Write a syntax extension to compile the protocols.

At some point, we'll need to add support for select.

*/

mod pingpong {
    enum ping = *pipes::packet<pong>;
    enum pong = *pipes::packet<ping>;

    fn init() -> (client::ping, server::ping) {
        pipes::entangle()
    }

    mod client {
        type ping = pipes::send_packet<pingpong::ping>;
        type pong = pipes::recv_packet<pingpong::pong>;

        fn do_ping(-c: ping) -> pong {
            let p = pipes::packet();

            pipes::send(c, pingpong::ping(p));
            pipes::recv_packet(p)
        }

        fn do_pong(-c: pong) -> (ping, ()) {
            let packet = pipes::recv(c);
            if packet == none {
                fail "sender closed the connection"
            }
            (pipes::send_packet(*option::unwrap(packet)), ())
        }
    }

    mod server {
        type ping = pipes::recv_packet<pingpong::ping>;
        type pong = pipes::send_packet<pingpong::pong>;

        fn do_ping(-c: ping) -> (pong, ()) {
            let packet = pipes::recv(c);
            if packet == none {
                fail "sender closed the connection"
            }
            (pipes::send_packet(*option::unwrap(packet)), ())
        }

        fn do_pong(-c: pong) -> ping {
            let p = pipes::packet();
            pipes::send(c, pingpong::pong(p));
            pipes::recv_packet(p)
        }
    }
}

fn client(-chan: pingpong::client::ping) {
    let chan = pingpong::client::do_ping(chan);
    log(error, "Sent ping");
    let (chan, _data) = pingpong::client::do_pong(chan);
    log(error, "Received pong");
}

fn server(-chan: pingpong::server::ping) {
    let (chan, _data) = pingpong::server::do_ping(chan);
    log(error, "Received ping");
    let chan = pingpong::server::do_pong(chan);
    log(error, "Sent pong");
}

fn main() {
    let (client_, server_) = pingpong::init();
    let client_ = ~mut some(client_);
    let server_ = ~mut some(server_);

    do task::spawn |move client_| {
        let mut client__ = none;
        *client_ <-> client__;
        client(option::unwrap(client__));
    };
    do task::spawn |move server_| {
        let mut server_ˊ = none;
        *server_ <-> server_ˊ;
        server(option::unwrap(server_ˊ));
    };
}
