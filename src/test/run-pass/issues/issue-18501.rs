// run-pass
// Test that we don't ICE when inlining a function from another
// crate that uses a trait method as a value due to incorrectly
// translating the def ID of the trait during AST decoding.

// aux-build:issue-18501.rs
// pretty-expanded FIXME #23616

extern crate issue_18501 as issue;

fn main() {
    issue::pass_method();
}
