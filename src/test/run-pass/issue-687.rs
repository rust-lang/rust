use std;
import std::vec;
import std::task;
import std::comm;
import std::comm::chan;
import std::comm::port;
import std::comm::recv;
import std::comm::send;

tag msg { closed; received([u8]); }

fn producer(c: chan<[u8]>) {
    send(c, [1u8, 2u8, 3u8, 4u8]);
    let empty: [u8] = [];
    send(c, empty);
}

fn packager(cb: chan<chan<[u8]>>, msg: chan<msg>) {
    let p: port<[u8]> = port();
    send(cb, chan(p));
    while true {
        log "waiting for bytes";
        let data = recv(p);
        log "got bytes";
        if vec::len(data) == 0u { log "got empty bytes, quitting"; break; }
        log "sending non-empty buffer of length";
        log vec::len(data);
        send(msg, received(data));
        log "sent non-empty buffer";
    }
    log "sending closed message";
    send(msg, closed);
    log "sent closed message";
}

fn main() {
    let p: port<msg> = port();
    let recv_reader: port<chan<[u8]>> = port();
    let pack =
        task::spawn(bind packager(chan(recv_reader), chan(p)));

    let source_chan: chan<[u8]> = recv(recv_reader);
    let prod = task::spawn(bind producer(source_chan));

    while true {
        let msg = recv(p);
        alt msg {
          closed. { log "Got close message"; break; }
          received(data) {
            log "Got data. Length is:";
            log vec::len::<u8>(data);
          }
        }
    }
}
