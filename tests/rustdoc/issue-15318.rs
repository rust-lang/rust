// aux-build:issue-15318.rs
// ignore-cross-compile

#![no_std]

extern crate issue_15318;

// @has issue_15318/fn.bar.html \
//      '//*[@href="http://example.com/issue_15318/primitive.pointer.html"]' \
//      '*mut T'
pub fn bar<T>(ptr: *mut T) {}
