// -*- rust -*-

// error-pattern:1 == 2
use std;
import task;
import comm::port;
import comm::recv;

fn child() { assert (1 == 2); }

fn main() {
    let p = port::<int>();
    task::spawn {|| child(); };
    let x = recv(p);
}
