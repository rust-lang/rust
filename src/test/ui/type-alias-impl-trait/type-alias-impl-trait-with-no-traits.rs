// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type Foo = impl 'static;
//~^ ERROR: at least one trait must be specified

fn foo() -> Foo {
    "foo"
}

fn bar() -> impl 'static { //~ ERROR: at least one trait must be specified
    "foo"
}

fn main() {}
