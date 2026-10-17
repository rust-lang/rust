//! Regression test for <https://github.com/rust-lang/rust/issues/16643>.
//! Tests that method which matches on self with for loop doesn't ICE cross-crate.

//@ run-pass
//@ aux-build:for-loop-in-match-on-self.rs


extern crate for_loop_in_match_on_self as i;

pub fn main() {
    i::TreeBuilder { h: 3 }.process_token();
}
