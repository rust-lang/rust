// -*- rust -*-

use std;
import comm;
import comm::send;
import comm::chan;
import comm::recv;
import task;

fn a(c: chan<int>) { #debug("task a0"); #debug("task a1"); send(c, 10); }

fn main() {
    let p = comm::port();
    task::spawn(chan(p), a);
    task::spawn(chan(p), b);
    let n: int = 0;
    n = recv(p);
    n = recv(p);
    #debug("Finished.");
}

fn b(c: chan<int>) {
    #debug("task b0");
    #debug("task b1");
    #debug("task b2");
    #debug("task b2");
    #debug("task b3");
    send(c, 10);
}
