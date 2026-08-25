// https://github.com/rust-lang/rust/issues/76736

//@ aux-build:issue-76736-1.rs
//@ aux-build:issue-76736-2.rs

// https://github.com/rust-lang/rust/issues/124635

#![crate_name = "foo"]
#![feature(rustc_private, staged_api)]
#![unstable(feature = "rustc_private", issue = "none")]

extern crate issue_76736_1;
extern crate issue_76736_2;

//@ has foo/struct.Foo.html
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'MaybeResult'
pub struct Foo;

//@ has foo/struct.Bar.html
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'MaybeResult'
pub use issue_76736_2::Bar;
