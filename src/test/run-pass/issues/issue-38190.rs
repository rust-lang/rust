// run-pass
// aux-build:issue_38190.rs
// ignore-pretty issue #37195

#[macro_use]
extern crate issue_38190;

mod auxiliary {
    m!([mod issue_38190;]);
}

fn main() {}
