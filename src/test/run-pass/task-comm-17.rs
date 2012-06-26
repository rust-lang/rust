// Issue #922

// This test is specifically about spawning temporary closures.

use std;
import task;

fn f() {
}

fn main() {
    task::spawn({|| f() });
}