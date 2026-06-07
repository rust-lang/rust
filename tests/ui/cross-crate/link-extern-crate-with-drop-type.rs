//@ run-pass
//@ aux-build:link-extern-crate-with-drop-type.rs
//! Regression test for https://github.com/rust-lang/rust/issues/2170
//! This test just verifies that linking against an external crate works without
//! a metadata failure. Apparently, having a Drop that calls another function is the trigger.

extern crate link_extern_crate_with_drop_type;

pub fn main() {
   // let _ = issue_2170_lib::rsrc(2);
}
