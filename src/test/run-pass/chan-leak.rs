// Issue #763

use std;
import std::task;
import std::comm::chan;
import std::comm::send;
import std::comm;
import std::comm::port;
import std::comm::recv;

tag request { quit; close(chan<bool>); }

type ctx = chan<request>;

fn# request_task(c: chan<ctx>) {
    let p = port();
    send(c, chan(p));
    let req: request;
    req = recv(p);
    // Need to drop req before receiving it again
    req = recv(p);
}

fn new() -> ctx {
    let p = port();
    let t = task::spawn2(chan(p), request_task);
    let cx: ctx;
    cx = recv(p);
    ret cx;
}

fn main() {
    let cx = new();

    let p = port::<bool>();
    send(cx, close(chan(p)));
    send(cx, quit);
}
