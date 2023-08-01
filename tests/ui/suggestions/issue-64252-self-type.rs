// This test checks that a suggestion to add a `self: ` parameter name is provided
// to functions where this is applicable.

pub fn foo(Box<Self>) { } //~ ERROR generic args in patterns require the turbofish syntax

struct Bar;

impl Bar {
    fn bar(Box<Self>) { } //~ ERROR generic args in patterns require the turbofish syntax
}

fn main() { }
