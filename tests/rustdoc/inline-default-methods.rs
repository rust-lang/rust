// aux-build:inline-default-methods.rs
// ignore-cross-compile

extern crate inline_default_methods;

// @has inline_default_methods/trait.Foo.html
// @has - '//pre[@class="rust item-decl"]' '// Required method fn bar(&self);'
// @has - '//pre[@class="rust item-decl"]' '// Provided method fn foo(&mut self)'
pub use inline_default_methods::Foo;

// @has inline_default_methods/trait.Bar.html
// @has - '//pre[@class="rust item-decl"]' '// Required method fn bar(&self);'
// @has - '//pre[@class="rust item-decl"]' '// Provided methods fn foo1(&mut self)'
// @has - '//pre[@class="rust item-decl"]' 'fn foo2(&mut self)'
pub use inline_default_methods::Bar;

// @has inline_default_methods/trait.Baz.html
// @has - '//pre[@class="rust item-decl"]' '// Required methods fn bar1(&self);'
// @has - '//pre[@class="rust item-decl"]' 'fn bar2(&self);'
// @has - '//pre[@class="rust item-decl"]' '// Provided method fn foo(&mut self)'
pub use inline_default_methods::Baz;
