// -*- rust -*-

// error-pattern:fail
use std;
import task;
import comm::Port;
import comm::recv;

fn grandchild() { fail ~"grandchild dies"; }

fn child() {
    let p = Port::<int>();
    task::spawn(|| grandchild() );
    let x = recv(p);
}

fn main() {
    let p = Port::<int>();
    task::spawn(|| child() );
    let x = recv(p);
}
