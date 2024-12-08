//@ run-rustfix

#![allow(warnings)]

// Test that suggestion to add `*` characters applies to implementations of `Deref` as well as
// references.

struct Foo(i32);

struct Bar(String);

impl std::ops::Deref for Foo {
    type Target = i32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Deref for Bar {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn main() {
    let x = Foo(42);
    let y: i32 = x; //~ ERROR mismatched types
    let a = &42;
    let b: i32 = a; //~ ERROR mismatched types

    // Do not make a suggestion when adding a `*` wouldn't actually fix the issue:
    let f = Bar("bar".to_string());
    let g: String = f; //~ ERROR mismatched types
}
