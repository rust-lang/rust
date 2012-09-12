// xfail-fast (aux-build)
// aux-build:issue_2242_a.rs
// aux-build:issue_2242_b.rs
// aux-build:issue_2242_c.rs

extern mod a;
extern mod b;
extern mod c;

use a::to_strz;

fn main() {
    io::println((~"foo").to_strz());
    io::println(1.to_strz());
    io::println(true.to_strz());
}
