// -*- rust -*-

// error-pattern:fail
use std;
use comm::Chan;
use comm::Port;
use comm::recv;

fn child() { fail; }

fn main() {
    let p = Port::<int>();
    task::spawn(|| child() );
    task::yield();
}
