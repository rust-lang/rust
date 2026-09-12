// If there is more than one dependency level, the transient crates reexports used to be
// skipped. This test ensures it's not the case anymore.
// Regression test <https://github.com/rust-lang/rust/issues/81893>.

//@ aux-build: transient-reexports.rs

#![crate_name = "foo"]

extern crate bar;

//@ has 'foo/index.html'

// We ensure that there is only one item, so it's not possibly another item we're gonna check.
//@ count - '//dl/[@class="item-table"]/dt' 1
//@ count - '//dl/[@class="item-table"]/dd' 1
//@ has - '//dl/[@class="item-table"]/dt/a[@href="struct.Type.html"]' 'Type'
// We should have "foo", "bar" and "baz" (one fragment in each crate reexport).
//@ has - '//dl/[@class="item-table"]/dd' 'foo bar baz'

/// foo
pub use bar::Type;
