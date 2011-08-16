// Issue #763

use std;
import std::task;
import std::comm::_chan;
import std::comm::send;
import std::comm;
import std::comm::mk_port;

tag request {
  quit;
  close(_chan<bool>);
}

type ctx = _chan<request>;

fn request_task(c: _chan<ctx>) {
    let p = mk_port();
    send(c, p.mk_chan());
    let req: request;
    req = p.recv();
    // Need to drop req before receiving it again
    req = p.recv();
}

fn new() -> ctx {
    let p = mk_port();
    let t = task::_spawn(bind request_task(p.mk_chan()));
    let cx: ctx;
    cx = p.recv();
    ret cx;
}

fn main() {
    let cx = new();

    let p = mk_port::<bool>();
    send(cx, close(p.mk_chan()));
    send(cx, quit);
}
