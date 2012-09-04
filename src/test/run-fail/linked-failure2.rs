// -*- rust -*-

// error-pattern:fail
use std;
import task;
import comm::Chan;
import comm::Port;
import comm::recv;

fn child() { fail; }

fn main() {
    let p = Port::<int>();
    task::spawn(|| child() );
    task::yield();
}
