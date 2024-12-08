//@ aux-build:rustdoc-hidden.rs
//@ build-aux-docs
//@ ignore-cross-compile

extern crate rustdoc_hidden;

//@ has inline_hidden/index.html
// Ensures this item is not inlined.
//@ has - '//*[@id="reexport.Foo"]/code' 'pub use rustdoc_hidden::Foo;'
#[doc(no_inline)]
pub use rustdoc_hidden::Foo;

// Even if the foreign item has `doc(hidden)`, we should be able to inline it.
//@ has - '//*[@class="item-name"]/a[@class="struct"]' 'Inlined'
#[doc(inline)]
pub use rustdoc_hidden::Foo as Inlined;

// Even with this import, we should not see `Foo`.
//@ count - '//*[@class="item-name"]' 4
//@ has - '//*[@class="item-name"]/a[@class="struct"]' 'Bar'
//@ has - '//*[@class="item-name"]/a[@class="fn"]' 'foo'
pub use rustdoc_hidden::*;

//@ has inline_hidden/fn.foo.html
//@ !has - '//a/@title' 'Foo'
pub fn foo(_: Foo) {}
