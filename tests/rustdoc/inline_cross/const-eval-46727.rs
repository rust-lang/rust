// https://github.com/rust-lang/rust/issues/46727
#![crate_name="foo"]

//@ aux-build:issue-46727.rs

extern crate issue_46727;

//@ has foo/trait.Foo.html
//@ has - '//h3[@class="code-header"]' 'impl<T> Foo for Bar<[T; 3]>'
pub use issue_46727::{Foo, Bar};
