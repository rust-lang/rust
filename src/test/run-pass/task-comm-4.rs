use std;
import std::comm;
import std::comm::send;

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::mk_port();
    let c = p.mk_chan();
    send(c, 1);
    send(c, 2);
    send(c, 3);
    send(c, 4);
    r = p.recv();
    sum += r;
    log r;
    r = p.recv();
    sum += r;
    log r;
    r = p.recv();
    sum += r;
    log r;
    r = p.recv();
    sum += r;
    log r;
    send(c, 5);
    send(c, 6);
    send(c, 7);
    send(c, 8);
    r = p.recv();
    sum += r;
    log r;
    r = p.recv();
    sum += r;
    log r;
    r = p.recv();
    sum += r;
    log r;
    r = p.recv();
    sum += r;
    log r;
    assert (sum == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}