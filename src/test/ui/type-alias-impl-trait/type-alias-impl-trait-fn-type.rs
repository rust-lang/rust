#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// FIXME: this is ruled out for now but should work

type Foo = fn() -> impl Send;
//~^ ERROR: `impl Trait` isn't allowed within `fn` pointer return [E0562]

fn make_foo() -> Foo {
    || 15
}

fn main() {}
