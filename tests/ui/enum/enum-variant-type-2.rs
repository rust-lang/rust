// Test that enum variants are not actually types.

enum Foo {
    Bar
}

fn foo(x: Foo::Bar) {} //~ ERROR expected type, found variant `Foo::Bar`

fn main() {}
