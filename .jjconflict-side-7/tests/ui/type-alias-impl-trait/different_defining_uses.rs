#![feature(type_alias_impl_trait)]

fn main() {}

// two definitions with different types
type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
fn foo() -> Foo {
    ""
}

#[define_opaque(Foo)]
fn bar() -> Foo {
    //~^ ERROR concrete type differs
    42i32
}
