//@ aux-build:impl-for-projection.rs

extern crate foo;

// FIXME: because rustdoc doesn't normalize types, it doesn't inline the impl in foo
// that is for a projection that resolves to `Struct`
pub use foo::Struct;
