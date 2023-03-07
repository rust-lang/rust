// run-pass
// aux-build:issue-16643.rs

// pretty-expanded FIXME #23616

extern crate issue_16643 as i;

pub fn main() {
    i::TreeBuilder { h: 3 }.process_token();
}
