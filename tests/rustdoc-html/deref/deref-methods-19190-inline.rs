//@ aux-build:issue-19190-3.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/19190
#![crate_name="issue_19190_3"]

extern crate issue_19190_3;

use std::ops::Deref;
use issue_19190_3::Baz;

//@ has issue_19190_3/struct.Foo.html
//@ has - '//*[@id="method.as_str"]' 'fn as_str(&self) -> &str'
//@ !has - '//*[@id="method.new"]' 'fn new() -> String'
pub use issue_19190_3::Foo;

//@ has issue_19190_3/struct.Bar.html
//@ has - '//*[@id="method.baz"]' 'fn baz(&self)'
//@ !has - '//*[@id="method.static_baz"]' 'fn static_baz()'
pub use issue_19190_3::Bar;

//@ has issue_19190_3/struct.MyBar.html
//@ has - '//*[@id="method.baz"]' 'fn baz(&self)'
//@ !has - '//*[@id="method.static_baz"]' 'fn static_baz()'
pub struct MyBar;

impl Deref for MyBar {
    type Target = Baz;
    fn deref(&self) -> &Baz { loop {} }
}
