import str::sbuf;
import task;

native "rust" mod rustrt {
    type socket;
    type server;
    fn aio_init();
    fn aio_run();
    fn aio_stop();
    fn aio_connect(host: sbuf, port: int, connected: chan[socket]);
    fn aio_serve(host: sbuf, port: int, acceptChan: chan[socket]) -> server;
    fn aio_writedata(s: socket, buf: *u8, size: uint, status: chan[bool]);
    fn aio_read(s: socket, reader: chan[[u8]]);
    fn aio_close_server(s: server, status: chan[bool]);
    fn aio_close_socket(s: socket);
    fn aio_is_null_client(s: socket) -> bool;
}

type server = rustrt::server;
type client = rustrt::socket;
tag pending_connection {
    remote(str,int);
    incoming(server);
}

tag socket_event {
    connected(client);
    closed;
    received([u8]);
}

tag server_event {
    pending(chan[chan[socket_event]]);
}

tag request {
    quit;
    connect(pending_connection,chan[socket_event]);
    serve(str,int,chan[server_event],chan[server]);
    write(client,[u8],chan[bool]);
    close_server(server, chan[bool]);
    close_client(client);
}

type ctx = chan[request];

fn connect_task(ip: str, portnum: int, evt: chan[socket_event]) {
    let connecter: port[client] = port();
    rustrt::aio_connect(str::buf(ip), portnum, chan(connecter));
    let client: client;
    connecter |> client;
    new_client(client, evt);
}

fn new_client(client: client, evt: chan[socket_event]) {
    // Start the read before notifying about the connect.  This avoids a race
    // condition where the receiver can close the socket before we start
    // reading.
    let reader: port[[u8]] = port();
    rustrt::aio_read(client, chan(reader));

    evt <| connected(client);

    while (true) {
        log "waiting for bytes";
        let data: [u8];
        reader |> data;
        log "got some bytes";
        log ivec::len[u8](data);
        if (ivec::len[u8](data) == 0u) {
            log "got empty buffer, bailing";
            break;
        }
        log "got non-empty buffer, sending";
        evt <| received(data);
        log "sent non-empty buffer";
    }
    log "done reading";
    evt <| closed;
    log "close message sent";
}

fn accept_task(client: client, events: chan[server_event]) {
    log "accept task was spawned";
    let p: port[chan[socket_event]] = port();
    events <| pending(chan(p));
    let evt: chan[socket_event];
    p |> evt;
    new_client(client, evt);
    log "done accepting";
}

fn server_task(ip: str, portnum: int, events: chan[server_event],
               server: chan[server]) {
    let accepter: port[client] = port();
    server <| rustrt::aio_serve(str::buf(ip), portnum, chan(accepter));

    let client: client;
    while (true) {
        log "preparing to accept a client";
        accepter |> client;
        if (rustrt::aio_is_null_client(client)) {
          log "client was actually null, returning";
          ret;
        } else {
          task::_spawn(bind accept_task(client, events));
        }
    }
}

fn request_task(c: chan[ctx]) {
    // Create a port to accept IO requests on
    let p: port[request] = port();
    // Hand of its channel to our spawner
    c <| chan(p);
    log "uv run task spawned";
    // Spin for requests
    let req: request;
    while (true) {
        p |> req;
        alt req {
            quit. {
                log "got quit message";

                log "stopping libuv";
                rustrt::aio_stop();
                ret;
            }
            connect(remote(ip,portnum),client) {
                task::_spawn(bind connect_task(ip, portnum, client));
            }
            serve(ip,portnum,events,server) {
                task::_spawn(bind server_task(ip, portnum, events, server));
            }
            write(socket,v,status) {
                rustrt::aio_writedata(socket,
                                      ivec::to_ptr[u8](v), ivec::len[u8](v),
                                      status);
            }
            close_server(server,status) {
                log "closing server";
                rustrt::aio_close_server(server,status);
            }
            close_client(client) {
                log "closing client";
                rustrt::aio_close_socket(client);
            }
        }
    }
}

fn iotask(c: chan[ctx]) {
    log "io task spawned";
    // Initialize before accepting requests
    rustrt::aio_init();

    log "io task init";
    // Spawn our request task
    let reqtask = task::_spawn(bind request_task(c));

    log "uv run task init";
    // Enter IO loop. This never returns until aio_stop is called.
    rustrt::aio_run();
    log "waiting for request task to finish";

    task::join_id(reqtask);
}

fn new() -> ctx {
    let p: port[ctx] = port();
    let t = task::_spawn(bind iotask(chan(p)));
    let cx: ctx;
    p |> cx;
    ret cx;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
