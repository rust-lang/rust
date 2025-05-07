//@ run-pass
// Reported as issue #126, child leaks the string.

//@ needs-threads

use std::thread;

fn child2(_s: String) {}

pub fn main() {
    let _x = thread::spawn(move || child2("hi".to_string()));
}
