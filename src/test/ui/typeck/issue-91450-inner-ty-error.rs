// Regression test for #91450.
// This test ensures that the compiler does not suggest `Foo<[type error]>` in diagnostic messages.

fn foo() -> Option<_> {} //~ ERROR: [E0308]
//~^ ERROR: the type placeholder `_` is not allowed

fn main() {}
