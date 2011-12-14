// xfail-test
// Issue #922

// This test is specifically about spawning temporary closures, which
// isn't possible under the bare-fn regime. I'm keeping it around
// until such time as we have unique closures.

use std;
import task;

fn f() {
}

fn main() {
    task::spawn(bind f());
}