// aux-build:issue-31702-1.rs
// aux-build:issue-31702-2.rs
// ignore-test: FIXME(#31702) when this test was added it was thought that the
//                            accompanying llvm update would fix it, but
//                            unfortunately it appears that was not the case. In
//                            the interest of not deleting the test, though,
//                            this is just tagged with ignore-test

// this test is actually entirely in the linked library crates

extern crate issue_31702_1;
extern crate issue_31702_2;

fn main() {}
