// Issue #922

// This test is specifically about spawning temporary closures.

use std;

fn f() {
}

fn main() {
    task::spawn(|| f() );
}