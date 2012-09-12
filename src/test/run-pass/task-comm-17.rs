// Issue #922

// This test is specifically about spawning temporary closures.

extern mod std;

fn f() {
}

fn main() {
    task::spawn(|| f() );
}