// -*- rust -*-
// error-pattern:1 == 2
use std;
import task;
import comm::Chan;
import comm::Port;
import comm::recv;

fn child() { assert (1 == 2); }

fn parent() {
    let p = Port::<int>();
    task::spawn(|| child() );
    let x = recv(p);
}

// This task is not linked to the failure chain, but since the other
// tasks are going to fail the kernel, this one will fail too
fn sleeper() {
    let p = Port::<int>();
    let x = recv(p);
}

fn main() {
    task::spawn(|| sleeper() );
    task::spawn(|| parent() );
}