// -*- rust -*-

use std;
import comm;
import comm::Port;
import comm::send;
import comm::Chan;
import comm::recv;
import task;

fn a(c: Chan<int>) { debug!("task a0"); debug!("task a1"); send(c, 10); }

fn main() {
    let p = Port();
    let ch = Chan(p);
    task::spawn(|| a(ch) );
    task::spawn(|| b(ch) );
    let mut n: int = 0;
    n = recv(p);
    n = recv(p);
    debug!("Finished.");
}

fn b(c: Chan<int>) {
    debug!("task b0");
    debug!("task b1");
    debug!("task b2");
    debug!("task b2");
    debug!("task b3");
    send(c, 10);
}
