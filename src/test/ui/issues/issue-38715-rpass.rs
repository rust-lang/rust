// run-pass
// aux-build:issue-38715.rs
// aux-build:issue-38715-modern.rs

// Test that `#[macro_export] macro_rules!` shadow earlier `#[macro_export] macro_rules!`

#[macro_use]
extern crate issue_38715;
#[macro_use]
extern crate issue_38715_modern;

fn main() {
    foo!();
    foo_modern!();
}
