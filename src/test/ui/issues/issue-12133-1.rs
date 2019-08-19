// run-pass
// aux-build:issue-12133-rlib.rs
// aux-build:issue-12133-dylib.rs

// pretty-expanded FIXME #23616

extern crate issue_12133_rlib as a;
extern crate issue_12133_dylib as b;

fn main() {}
