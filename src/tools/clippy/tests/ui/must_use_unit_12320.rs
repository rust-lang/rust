//@aux-build:proc_macros.rs
//@no-rustfix

#![warn(clippy::must_use_unit)]
#![allow(clippy::unused_unit)]

#[cfg_attr(all(), must_use, deprecated)]
fn issue_12320() {}

#[cfg_attr(all(), deprecated, doc = "foo", must_use)]
fn issue_12320_2() {}
