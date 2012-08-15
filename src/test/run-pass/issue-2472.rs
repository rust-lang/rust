// aux-build:issue_2472_b.rs

use issue_2472_b;

import issue_2472_b::{S, T};

fn main() {
    let s = S(());
    s.foo();
    s.bar();
}
