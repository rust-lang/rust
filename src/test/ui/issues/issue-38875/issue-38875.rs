// aux-build:issue-38875-b.rs
// build-pass (FIXME(62277): could be check-pass?)

extern crate issue_38875_b;

fn main() {
    let test_x = [0; issue_38875_b::FOO];
}
