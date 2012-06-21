// xfail-test
// aux-build:issue_2242_a.rs
// aux-build:issue_2242_b.rs
// aux-build:issue_2242_c.rs

use a;
use b;
use c;

import b::to_str;
import c::to_str;

fn main() {
    io::println("foo".to_str());
    io::println(1.to_str());
    io::println(true.to_str());
}
