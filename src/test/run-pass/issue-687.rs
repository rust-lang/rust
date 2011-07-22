// xfail-stage0

use std;
import std::ivec;

tag msg {
    closed;
    received(u8[]);
}

fn producer(chan[u8[]] c) {
    c <| ~[1u8, 2u8, 3u8, 4u8];
    let u8[] empty = ~[];
    c <| empty;
}

fn packager(chan[chan[u8[]]] cb, chan[msg] msg) {
    let port[u8[]] p = port();
    cb <| chan(p);
    while (true) {
        log "waiting for bytes";
        let u8[] data;
        p |> data;
        log "got bytes";
        if (ivec::len[u8](data) == 0u) {
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
    let port[msg] p = port();
    let port[chan[u8[]]] recv_reader = port();
    auto pack = spawn packager(chan(recv_reader), chan(p));

    let chan[u8[]] source_chan;
    recv_reader |> source_chan;
    let task prod = spawn producer(source_chan);

    while (true) {
        let msg msg;
        p |> msg;
        alt (msg) {
            case (closed) {
                log "Got close message";
                break;
            }
            case (received(?data)) {
                log "Got data. Length is:";
                log ivec::len[u8](data);
            }
        }
    }
}
