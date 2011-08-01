// xfail-stage0
// xfail-pretty

use std;
import std::ivec;

tag msg { closed; received(u8[]); }

fn producer(c: chan[u8[]]) {
    c <| ~[1u8, 2u8, 3u8, 4u8];
    let empty: u8[] = ~[];
    c <| empty;
}

fn packager(cb: chan[chan[u8[]]], msg: chan[msg]) {
    let p: port[u8[]] = port();
    cb <| chan(p);
    while true {
        log "waiting for bytes";
        let data: u8[];
        p |> data;
        log "got bytes";
        if ivec::len[u8](data) == 0u {
            log "got empty bytes, quitting";
            break;
        }
        log "sending non-empty buffer of length";
        log ivec::len[u8](data);
        msg <| received(data);
        log "sent non-empty buffer";
    }
    log "sending closed message";
    msg <| closed;
    log "sent closed message";
}

fn main() {
    let p: port[msg] = port();
    let recv_reader: port[chan[u8[]]] = port();
    let pack = spawn packager(chan(recv_reader), chan(p));

    let source_chan: chan[u8[]];
    recv_reader |> source_chan;
    let prod: task = spawn producer(source_chan);


    while true {
        let msg: msg;
        p |> msg;
        alt msg {
          closed. { log "Got close message"; break; }
          received(data) {
            log "Got data. Length is:";
            log ivec::len[u8](data);
          }
        }
    }
}