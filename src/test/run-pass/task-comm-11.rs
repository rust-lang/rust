// xfail-stage3
use std;
import std::comm;
import std::task;

fn start(c: comm::_chan<comm::_chan<int>>) {
    let p : comm::_port<int> = comm::mk_port();
    comm::send(c, p.mk_chan());
}

fn main() {
    let p = comm::mk_port<comm::_chan<int>>();
    let child = task::_spawn(bind start(p.mk_chan()));
    let c = p.recv();
}
