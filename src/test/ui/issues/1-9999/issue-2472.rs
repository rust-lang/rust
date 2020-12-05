// run-pass
// aux-build:issue-2472-b.rs

// pretty-expanded FIXME #23616

extern crate issue_2472_b;

use issue_2472_b::{S, T};

pub fn main() {
    let s = S(());
    s.foo();
    s.bar();
}
