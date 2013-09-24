// Test that we print out the names of type parameters correctly in
// our error messages.

fn foo<Foo, Bar>(x: Foo) -> Bar { x } //~ ERROR expected `Bar` but found `Foo`

fn main() {}
