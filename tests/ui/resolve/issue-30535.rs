// aux-build:issue-30535.rs

extern crate issue_30535 as foo;

fn bar(
    _: foo::Foo::FooV //~ ERROR expected type, found variant `foo::Foo::FooV`
) {}

fn main() {}
