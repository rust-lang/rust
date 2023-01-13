// run-pass
// aux-build:issue-38190.rs
// ignore-pretty issue #37195

#[macro_use]
extern crate issue_38190;

mod auxiliary {
    m!([
        #[path = "issue-38190.rs"]
        mod issue_38190;
    ]);
}

fn main() {}
