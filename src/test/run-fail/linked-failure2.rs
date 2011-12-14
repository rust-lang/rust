// -*- rust -*-

// error-pattern:fail
use std;
import task;
import comm::chan;
import comm::port;
import comm::recv;

fn child(&&_i: ()) { fail; }

fn main() {
    let p = port::<int>();
    task::spawn((), child);
    task::yield();
}
