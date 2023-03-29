#![feature(type_alias_impl_trait)]

fn main() {}

// two definitions with different types
type Foo = impl std::fmt::Debug;

#[defines(Foo)]
fn foo() -> Foo {
    ""
}

#[defines(Foo)]
fn bar() -> Foo {
    42i32
    //~^ ERROR concrete type differs from previous
}
