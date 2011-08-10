import comm::_port;
import comm::_chan;
import comm::mk_port;
import comm::send;

import str;
import net;

type ctx = aio::ctx;
type client = { ctx: ctx, client: aio::client,
               evt: _port<aio::socket_event> };
type server = { ctx: ctx, server: aio::server,
               evt: _port<aio::server_event> };

fn new() -> ctx {
    ret aio::new();
}

fn destroy(ctx: ctx) {
    send(ctx, aio::quit);
}

fn make_socket(ctx: ctx, p: _port<aio::socket_event>) -> client {
    let evt: aio::socket_event = p.recv();
    alt evt {
      aio::connected(client) {
        ret { ctx: ctx, client: client, evt: p };
      }
      _ { fail "Could not connect to client"; }
    }
}

fn connect_to(ctx: ctx, ip: net::ip_addr, portnum: int) -> client {
    let p: _port<aio::socket_event> = mk_port();
    send(ctx, aio::connect(aio::remote(ip, portnum), p.mk_chan()));
    ret make_socket(ctx, p);
}

fn read(c: client) -> [u8] {
    alt c.evt.recv() {
        aio::closed. {
            ret ~[];
        }
        aio::received(buf) {
            ret buf;
        }
    }
}

fn create_server(ctx: ctx, ip: net::ip_addr, portnum: int) -> server {
    let evt: _port<aio::server_event> = mk_port();
    let p: _port<aio::server> = mk_port();
    send(ctx, aio::serve(ip, portnum,
                         evt.mk_chan(), p.mk_chan()));
    let srv: aio::server = p.recv();
    ret { ctx: ctx, server: srv, evt: evt };
}

fn accept_from(server: server) -> client {
    let evt: aio::server_event = server.evt.recv();
    alt evt {
      aio::pending(callback) {
        let p: _port<aio::socket_event> = mk_port();
        send(callback, p.mk_chan());
        ret make_socket(server.ctx, p);
      }
    }
}

fn write_data(c: client, data: [u8]) -> bool {
    let p: _port<bool> = mk_port();
    send(c.ctx, aio::write(c.client, data, p.mk_chan()));
    ret p.recv();
}

fn close_server(server: server) {
    // TODO: make this unit once we learn to send those from native code
    let p: _port<bool> = mk_port();
    send(server.ctx, aio::close_server(server.server, p.mk_chan()));
    log "Waiting for close";
    p.recv();
    log "Got close";
}

fn close_client(client: client) {
    send(client.ctx, aio::close_client(client.client));
    let evt: aio::socket_event;
    do {
        evt = client.evt.recv();
        alt evt {
          aio::closed. {
            ret;
          }
          _ {}
        }
    } while (true);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
