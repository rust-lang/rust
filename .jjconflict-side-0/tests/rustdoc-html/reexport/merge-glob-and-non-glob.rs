// This test ensures that if an item is inlined from two different `use`,
// then it will use attributes from both of them.
// This is a regression test for <https://github.com/rust-lang/rust/issues/143107>.

#![feature(no_core)]
#![no_core]
#![no_std]
#![crate_name = "foo"]

// First we ensure we only have two items.
//@ has 'foo/index.html'
//@ count - '//dl[@class="item-table"]/dt' 2
// We should also only have one section (Structs).
//@ count - '//h2[@class="section-header"]' 1
// We now check the short docs.
//@ has - '//dl[@class="item-table"]/dd' 'Foobar Blob'
//@ has - '//dl[@class="item-table"]/dd' 'Tarte Tatin'

//@ has 'foo/struct.Foo.html'
//@ has - '//*[@class="docblock"]' 'Foobar Blob'

//@ has 'foo/struct.Another.html'
//@ has - '//*[@class="docblock"]' 'Tarte Tatin'

mod raw {
    /// Blob
    pub struct Foo;

    /// Tatin
    pub struct Another;
}

/// Foobar
pub use raw::Foo;

// Glob reexport attributes are ignored.
/// Baz
pub use raw::*;

/// Tarte
pub use raw::Another as Another;
