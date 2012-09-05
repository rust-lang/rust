// aux-build:issue_2472_b.rs
// xfail-fast

use issue_2472_b;

use issue_2472_b::{S, T};

fn main() {
    let s = S(());
    s.foo();
    s.bar();
}
