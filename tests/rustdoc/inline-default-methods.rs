// aux-build:inline-default-methods.rs
// ignore-cross-compile

extern crate inline_default_methods;

// @has inline_default_methods/trait.Foo.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'fn bar(&self);'
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'fn foo(&mut self) { ... }'
pub use inline_default_methods::Foo;
