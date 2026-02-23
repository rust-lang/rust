// https://github.com/rust-lang/rust/issues/76736

//@ compile-flags: -Zforce-unstable-if-unmarked
//@ aux-build:issue-76736-1.rs
//@ aux-build:issue-76736-2.rs

#![crate_name = "foo"]

extern crate issue_76736_1;
extern crate issue_76736_2;

//@ has foo/struct.Foo.html
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'MaybeResult'
pub struct Foo;

//@ has foo/struct.Bar.html
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'MaybeResult'
pub use issue_76736_2::Bar;
