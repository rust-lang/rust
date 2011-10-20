use std;

import std::comm;
import std::comm::chan;
import std::comm::send;
import std::task;

fn main() { test05(); }

fn# test05_start(ch : chan<int>) {
    log_err ch;
    send(ch, 10);
    log_err "sent 10";
    send(ch, 20);
    log_err "sent 20";
    send(ch, 30);
    log_err "sent 30";
}

fn test05() {
    let po = comm::port();
    let ch = comm::chan(po);
    task::spawn(ch, test05_start);
    let value = comm::recv(po);
    log_err value;
    value = comm::recv(po);
    log_err value;
    value = comm::recv(po);
    log_err value;
    assert (value == 30);
}
