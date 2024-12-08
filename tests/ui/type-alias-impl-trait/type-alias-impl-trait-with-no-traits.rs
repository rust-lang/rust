#![feature(type_alias_impl_trait)]

type Foo = impl 'static;
//~^ ERROR: at least one trait must be specified

fn foo() -> Foo {
    "foo"
}

fn bar() -> impl 'static { //~ ERROR: at least one trait must be specified
    "foo"
}

fn main() {}
