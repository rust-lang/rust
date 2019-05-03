#![feature(existential_type)]

existential type Foo: 'static;
//~^ ERROR: at least one trait must be specified

fn foo() -> Foo {
    "foo"
}

fn bar() -> impl 'static { //~ ERROR: at least one trait must be specified
    "foo"
}

fn main() {}
