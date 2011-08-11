// Temporarily xfailing, because something is wrong.
// xfail-stage2
use std;

import std::comm;
import std::comm::chan_t;
import std::comm::send;
import std::task;

fn main() { test05(); }

fn test05_start(ch : chan_t[int]) {
    log_err ch;
    send(ch, 10);
    log_err "sent 10";
    send(ch, 20);
    log_err "sent 20";
    send(ch, 30);
    log_err "sent 30";
}

fn test05() {
    let po = comm::mk_port[int]();
    let ch = po.mk_chan2();
    task::_spawn(bind test05_start(ch));
    let value = po.recv();
    log_err value;
    value = po.recv();
    log_err value;
    value = po.recv();
    log_err value;
    assert (value == 30);
}
