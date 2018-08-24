// aux-build:issue-13560-1.rs
// aux-build:issue-13560-2.rs
// aux-build:issue-13560-3.rs

// Regression test for issue #13560, the test itself is all in the dependent
// libraries. The fail which previously failed to compile is the one numbered 3.

extern crate issue_13560_2 as t2;
extern crate issue_13560_3 as t3;

fn main() {}
