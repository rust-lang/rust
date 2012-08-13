// xfail-fast (aux-build)
// aux-build:issue_2242_a.rs
// aux-build:issue_2242_b.rs
// aux-build:issue_2242_c.rs

use a;
use b;
use c;

import a::to_strz;

fn main() {
    io::println((~"foo").to_strz());
    io::println(1.to_strz());
    io::println(true.to_strz());
}
