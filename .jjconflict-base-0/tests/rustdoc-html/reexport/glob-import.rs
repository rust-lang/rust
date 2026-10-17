// This test ensures that inlining a foreign crate through a glob import doesn't
// panic (because we're trying to retrieve attributes from an item which doesn't
// have attributes).
// Regression test for <https://github.com/rust-lang/rust/issues/158686>.

//@ aux-build: glob-import.rs

#![crate_name = "foo"]

extern crate glob_import;

//@ has 'foo/index.html'
// There should be only one item listed: `wgc` (the only `glob-import` dep public item).
//@ count - '//*[@id="main-content"]/h2[@class="section-header"]' 1
//@ count - '//*[@id="main-content"]/dl[@class="item-table"]/dt' 1

pub use glob_import::*;
