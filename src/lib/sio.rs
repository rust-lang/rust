type ctx = aio::ctx;
type client = { ctx: ctx, client: aio::client, evt: port[aio::socket_event] };
type server = { ctx: ctx, server: aio::server, evt: port[aio::server_event] };

fn new() -> ctx {
    ret aio::new();
}

fn destroy(ctx: ctx) {
    ctx <| aio::quit;
}

fn make_socket(ctx: ctx, p: port[aio::socket_event]) -> client {
    let evt: aio::socket_event;
    p |> evt;
    alt evt {
      aio::connected(client) {
        ret { ctx: ctx, client: client, evt: p };
      }
    }
    log_err ("Could not connect to client");
    fail;
}

fn connect_to(ctx: ctx, ip: str, portnum: int) -> client {
    let p: port[aio::socket_event] = port();
    ctx <| aio::connect(aio::remote(ip, portnum), chan(p));
    ret make_socket(ctx, p);
}

fn read(c: client) -> [u8] {
    let evt: aio::socket_event;
    c.evt |> evt;
    alt evt {
        aio::closed. {
            ret ~[];
        }
        aio::received(buf) {
            ret buf;
        }
    }
}

fn create_server(ctx: ctx, ip: str, portnum: int) -> server {
    let evt: port[aio::server_event] = port();
    let p: port[aio::server] = port();
    ctx <| aio::serve(ip, portnum, chan(evt), chan(p));
    let srv: aio::server;
    p |> srv;
    ret { ctx: ctx, server: srv, evt: evt };
}

fn accept_from(server: server) -> client {
    let evt: aio::server_event;
    server.evt |> evt;
    alt evt {
        aio::pending(callback) {
            let p: port[aio::socket_event] = port();
            callback <| chan(p);
            ret make_socket(server.ctx, p);
        }
    }
}

fn write_data(c: client, data: [u8]) -> bool {
    let p: port[bool] = port();
    c.ctx <| aio::write(c.client, data, chan(p));
    let success: bool;
    p |> success;
    ret success;
}

fn close_server(server: server) {
    // TODO: make this unit once we learn to send those from native code
    let p: port[bool] = port();
    server.ctx <| aio::close_server(server.server, chan(p));
    let success: bool;
    log "Waiting for close";
    p |> success;
    log "Got close";
}

fn close_client(client: client) {
    client.ctx <| aio::close_client(client.client);
    let evt: aio::socket_event;
    do {
        client.evt |> evt;
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
