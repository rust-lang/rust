// aux-build:issue-28927-2.rs
// aux-build:issue-28927-1.rs
// ignore-cross-compile

extern crate issue_28927_1 as inner1;
pub use inner1 as foo;
