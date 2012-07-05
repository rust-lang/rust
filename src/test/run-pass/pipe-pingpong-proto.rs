// xfail-test

// An example to make sure the protocol parsing syntax extension works.

proto! pingpong {
    ping:send {
        ping -> pong
    }

    pong:recv {
        pong -> ping
    }
}

fn main() {
    // TODO: do something with the protocol
}