// compile-pass
// aux-build:issue-57264.rs

extern crate issue_57264;

fn main() {
    issue_57264::Pub::pub_method();
}
