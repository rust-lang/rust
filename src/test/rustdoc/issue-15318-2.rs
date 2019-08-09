// aux-build:issue-15318.rs
// ignore-cross-compile

extern crate issue_15318;

pub use issue_15318::ptr;

// @has issue_15318_2/fn.bar.html \
//          '//*[@href="primitive.pointer.html"]' \
//          '*mut T'
pub fn bar<T>(ptr: *mut T) {}
