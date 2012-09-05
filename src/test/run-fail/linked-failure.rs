// -*- rust -*-

// error-pattern:1 == 2
use std;
use comm::Port;
use comm::recv;

fn child() { assert (1 == 2); }

fn main() {
    let p = Port::<int>();
    task::spawn(|| child() );
    let x = recv(p);
}
