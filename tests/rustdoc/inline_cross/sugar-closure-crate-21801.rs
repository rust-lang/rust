//@ aux-build:issue-21801.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/21801
#![crate_name="issue_21801"]

extern crate issue_21801;

//@ has issue_21801/struct.Foo.html
//@ has - '//*[@id="method.new"]' \
//        'fn new<F>(f: F) -> Foowhere F: FnMut() -> i32'
pub use issue_21801::Foo;
