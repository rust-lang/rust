// run-pass
#![allow(unused_imports)]
// aux-build:issue-12612-1.rs
// aux-build:issue-12612-2.rs

// pretty-expanded FIXME #23616

extern crate issue_12612_1 as foo;
extern crate issue_12612_2 as bar;

mod test {
    use bar::baz;
}

fn main() {}
