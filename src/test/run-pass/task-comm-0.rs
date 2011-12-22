use std;

import comm;
import comm::chan;
import comm::send;
import task;

fn main() { test05(); }

fn test05_start(ch : chan<int>) {
    log_full(core::error, ch);
    send(ch, 10);
    #error("sent 10");
    send(ch, 20);
    #error("sent 20");
    send(ch, 30);
    #error("sent 30");
}

fn test05() {
    let po = comm::port();
    let ch = comm::chan(po);
    task::spawn(ch, test05_start);
    let value = comm::recv(po);
    log_full(core::error, value);
    value = comm::recv(po);
    log_full(core::error, value);
    value = comm::recv(po);
    log_full(core::error, value);
    assert (value == 30);
}
