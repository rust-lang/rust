/*

The first test case using pipes. The idea is to break this into
several stages for prototyping. Here's the plan:

1. Write an already-compiled protocol using existing ports and chans.

2. Take the already-compiled version and add the low-level
synchronization code instead.

3. Write a syntax extension to compile the protocols.

At some point, we'll need to add support for select.

*/

mod pingpong {
    import newcomm::*;

    type pingpong = ~mut option<(chan<()>, port<()>)>;

    fn init() -> (client::ping, server::ping) {
        let cp = port();
        let sp = port();
        let cc = chan(sp);
        let sc = chan(cp);

        let client = client::ping(~mut some((cc, cp)));
        let server = server::ping(~mut some((sc, sp)));

        (client, server)
    }

    mod client {
        enum ping = pingpong;
        enum pong = pingpong;

        fn do_ping(-c: ping) -> pong {
            let mut op = none;
            op <-> **c;
            let (c, s) <- option::unwrap(op);
            c.send(());
            let p <- (c, s);
            pong(~mut some(p))
        }

        fn do_pong(-c: pong) -> (ping, ()) {
            let mut op = none;
            op <-> **c;
            let (c, s) <- option::unwrap(op);
            let d = s.recv();
            let p <- (c, s);
            (ping(~mut some(p)), d)
        }
    }

    mod server {
        enum ping = pingpong;
        enum pong = pingpong;

        fn do_ping(-c: ping) -> (pong, ()) {
            let mut op = none;
            op <-> **c;
            let (c, s) <- option::unwrap(op);
            let d = s.recv();
            let p <- (c, s);
            (pong(~mut some(p)), d)
        }

        fn do_pong(-c: pong) -> ping {
            let mut op = none;
            op <-> **c;
            let (c, s) <- option::unwrap(op);
            c.send(());
            let p <- (c, s);
            ping(~mut some(p))
        }
    }
}

fn client(-chan: pingpong::client::ping) {
    let chan = pingpong::client::do_ping(chan);
    log(error, "Sent ping");
    let (_chan, _data) = pingpong::client::do_pong(chan);
    log(error, "Received pong");
}

fn server(-chan: pingpong::server::ping) {
    let (chan, _data) = pingpong::server::do_ping(chan);
    log(error, "Received ping");
    let _chan = pingpong::server::do_pong(chan);
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
