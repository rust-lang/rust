use std;
import std::comm;

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::mk_port();
    let c = p.mk_chan();
    c.send(1);
    c.send(2);
    c.send(3);
    c.send(4);
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
    c.send(5);
    c.send(6);
    c.send(7);
    c.send(8);
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