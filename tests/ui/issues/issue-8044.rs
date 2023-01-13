// run-pass
// aux-build:issue-8044.rs

// pretty-expanded FIXME #23616

extern crate issue_8044 as minimal;
use minimal::{BTree, leaf};

pub fn main() {
    BTree::<isize> { node: leaf(1) };
}
