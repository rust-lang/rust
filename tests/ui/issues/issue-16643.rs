//@ run-pass
//@ aux-build:issue-16643.rs


extern crate issue_16643 as i;

pub fn main() {
    i::TreeBuilder { h: 3 }.process_token();
}
