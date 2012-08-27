// -*- rust -*-

// error-pattern:1 == 2
use std;
import task;
import comm::Port;
import comm::recv;

fn child() { assert (1 == 2); }

fn main() {
    let p = Port::<int>();
    task::spawn(|| child() );
    let x = recv(p);
}
