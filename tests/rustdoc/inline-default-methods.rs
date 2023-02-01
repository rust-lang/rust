// aux-build:inline-default-methods.rs
// ignore-cross-compile

extern crate inline_default_methods;

// @has inline_default_methods/trait.Foo.html
// @has - '//pre[@class="rust item-decl"]' 'fn bar(&self);'
// @has - '//pre[@class="rust item-decl"]' 'fn foo(&mut self) { ... }'
pub use inline_default_methods::Foo;
