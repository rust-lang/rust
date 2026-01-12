//@ aux-build:issue-21092.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/21092
#![crate_name="issue_21092"]

extern crate issue_21092;

//@ has issue_21092/struct.Bar.html
//@ has - '//*[@id="associatedtype.Bar"]' 'type Bar = i32'
pub use issue_21092::{Foo, Bar};
