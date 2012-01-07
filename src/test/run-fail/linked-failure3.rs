// -*- rust -*-

// error-pattern:fail
use std;
import task;
import comm::port;
import comm::recv;

fn grandchild() { fail "grandchild dies"; }

fn child() {
    let p = port::<int>();
    task::spawn {|| grandchild(); };
    let x = recv(p);
}

fn main() {
    let p = port::<int>();
    task::spawn {|| child(); };
    let x = recv(p);
}
