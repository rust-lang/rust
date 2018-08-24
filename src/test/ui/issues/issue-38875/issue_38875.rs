// aux-build:issue_38875_b.rs
// compile-pass

extern crate issue_38875_b;

fn main() {
    let test_x = [0; issue_38875_b::FOO];
}
