// aux-build:intra-doc-basic.rs
// build-aux-docs
extern crate a;

// @has 'basic/struct.Bar.html' '//a[@href="../a/struct.Foo.html"]' 'Foo'
pub use a::Bar;
