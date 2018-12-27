// run-pass
#![allow(unused_imports)]
// aux-build:issue_12612_1.rs
// aux-build:issue_12612_2.rs

// pretty-expanded FIXME #23616

extern crate issue_12612_1 as foo;
extern crate issue_12612_2 as bar;

mod test {
    use bar::baz;
}

fn main() {}
