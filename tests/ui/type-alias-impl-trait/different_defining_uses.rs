#![feature(type_alias_impl_trait)]

fn main() {}

// two definitions with different types
type Foo = impl std::fmt::Debug;

#[define_opaques(Foo)]
fn foo() -> Foo {
    ""
}

#[define_opaques(Foo)]
fn bar() -> Foo {
    42i32
    //~^ ERROR concrete type differs from previous
}
