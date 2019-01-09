// aux-build:issue-21801.rs
// ignore-cross-compile

extern crate issue_21801;

// @has issue_21801/struct.Foo.html
// @has - '//*[@id="method.new"]' \
//        'fn new<F>(f: F) -> Foo where F: FnMut() -> i32'
pub use issue_21801::Foo;
