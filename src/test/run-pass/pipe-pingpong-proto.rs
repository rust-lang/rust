// An example to make sure the protocol parsing syntax extension works.

// xfail-pretty

proto! pingpong (
    ping:send {
        ping -> pong
    }

    pong:recv {
        pong -> ping
    }
)

mod test {
    use pipes::recv;
    use pingpong::{ping, pong};

    fn client(-chan: pingpong::client::ping) {
        use pingpong::client;

        let chan = client::ping(chan);
        log(error, ~"Sent ping");
        let pong(_chan) = recv(chan);
        log(error, ~"Received pong");
    }
    
    fn server(-chan: pingpong::server::ping) {
        use pingpong::server;

        let ping(chan) = recv(chan);
        log(error, ~"Received ping");
        let _chan = server::pong(chan);
        log(error, ~"Sent pong");
    }
}

fn main() {
    let (client_, server_) = pingpong::init();
    let client_ = ~mut Some(client_);
    let server_ = ~mut Some(server_);

    do task::spawn |move client_| {
        let mut client__ = None;
        *client_ <-> client__;
        test::client(option::unwrap(client__));
    };
    do task::spawn |move server_| {
        let mut server_ˊ = None;
        *server_ <-> server_ˊ;
        test::server(option::unwrap(server_ˊ));
    };
}
