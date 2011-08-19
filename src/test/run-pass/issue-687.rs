use std;
import std::vec;
import std::task;
import std::comm;
import std::comm::_chan;
import std::comm::_port;
import std::comm::mk_port;
import std::comm::send;

tag msg { closed; received([u8]); }

fn producer(c: _chan<[u8]>) {
    send(c, [1u8, 2u8, 3u8, 4u8]);
    let empty: [u8] = [];
    send(c, empty);
}

fn packager(cb: _chan<_chan<[u8]>>, msg: _chan<msg>) {
    let p: _port<[u8]> = mk_port();
    send(cb, p.mk_chan());
    while true {
        log "waiting for bytes";
        let data = p.recv();
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
    let p: _port<msg> = mk_port();
    let recv_reader: _port<_chan<[u8]>> = mk_port();
    let pack =
        task::_spawn(bind packager(recv_reader.mk_chan(), p.mk_chan()));

    let source_chan: _chan<[u8]> = recv_reader.recv();
    let prod = task::_spawn(bind producer(source_chan));

    while true {
        let msg = p.recv();
        alt msg {
          closed. { log "Got close message"; break; }
          received(data) {
            log "Got data. Length is:";
            log vec::len::<u8>(data);
          }
        }
    }
}
