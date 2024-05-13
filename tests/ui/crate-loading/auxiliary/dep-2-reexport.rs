//@ edition:2021
//@ aux-build:multiple-dep-versions-2.rs
extern crate dependency;
pub use dependency::do_something;
