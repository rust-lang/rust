//@ aux-build:issue-15318.rs
//@ ignore-cross-compile
// https://github.com/rust-lang/rust/issues/15318

#![crate_name="issue_15318_2"]
#![no_std]

extern crate issue_15318;

//@ !has issue_15318_2/fn.bar.html \
//          '//*[@href="primitive.pointer.html"]' \
//          '*mut T'
pub fn bar<T>(ptr: *mut T) {}
