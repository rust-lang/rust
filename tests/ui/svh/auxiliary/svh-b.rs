//! This is a client of the `a` crate defined in `svn-a-base.rs`. The
//! rpass and cfail tests (such as `run-pass/svh-add-comment.rs`) use
//! it by swapping in a different object code library crate built from
//! some variant of `svn-a-base.rs`, and then we are checking if the
//! compiler properly ignores or accepts the change, based on whether
//! the change could affect the downstream crate content or not
//! (#14132).

#![crate_name = "b"]

extern crate a;

pub fn foo() { assert_eq!(a::foo::<()>(0), 3); }
