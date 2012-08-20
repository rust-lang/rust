// Ping-pong is a bounded protocol. This is place where I can
// experiment with what code the compiler should generate for bounded
// protocols.

// xfail-pretty

// This was generated initially by the pipe compiler, but it's been
// modified in hopefully straightforward ways.
mod pingpong {
    import pipes::*;

    type packets = {
        // This is probably a resolve bug, I forgot to export packet,
        // but since I didn't import pipes::*, it worked anyway.
        ping: packet<ping>,
        pong: packet<pong>,
    };

    fn init() -> (client::ping, server::ping) {
        let buffer = ~{
            header: buffer_header(),
            data: {
                ping: mk_packet::<ping>(),
                pong: mk_packet::<pong>()
            }
        };
        do pipes::entangle_buffer(buffer) |buffer, data| {
            data.ping.set_buffer(buffer);
            data.pong.set_buffer(buffer);
            ptr::addr_of(data.ping)
        }
    }
    enum ping = server::pong;
    enum pong = client::ping;
    mod client {
        fn ping(+pipe: ping) -> pong {
            {
                let b = pipe.reuse_buffer();
                let s = send_packet_buffered(ptr::addr_of(b.buffer.data.pong));
                let c = recv_packet_buffered(ptr::addr_of(b.buffer.data.pong));
                let message = pingpong::ping(s);
                pipes::send(pipe, message);
                c
            }
        }
        type ping = pipes::send_packet_buffered<pingpong::ping,
        pingpong::packets>;
        type pong = pipes::recv_packet_buffered<pingpong::pong,
        pingpong::packets>;
    }
    mod server {
        type ping = pipes::recv_packet_buffered<pingpong::ping,
        pingpong::packets>;
        fn pong(+pipe: pong) -> ping {
            {
                let b = pipe.reuse_buffer();
                let s = send_packet_buffered(ptr::addr_of(b.buffer.data.ping));
                let c = recv_packet_buffered(ptr::addr_of(b.buffer.data.ping));
                let message = pingpong::pong(s);
                pipes::send(pipe, message);
                c
            }
        }
        type pong = pipes::send_packet_buffered<pingpong::pong,
        pingpong::packets>;
    }
}

mod test {
    import pipes::recv;
    import pingpong::{ping, pong};

    fn client(-chan: pingpong::client::ping) {
        import pingpong::client;

        let chan = client::ping(chan); return;
        log(error, "Sent ping");
        let pong(_chan) = recv(chan);
        log(error, "Received pong");
    }
    
    fn server(-chan: pingpong::server::ping) {
        import pingpong::server;

        let ping(chan) = recv(chan); return;
        log(error, "Received ping");
        let _chan = server::pong(chan);
        log(error, "Sent pong");
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
