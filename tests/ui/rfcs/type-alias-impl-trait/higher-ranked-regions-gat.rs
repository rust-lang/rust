// Regression test for #97098.

#![feature(type_alias_impl_trait)]

pub trait Trait {
    type Assoc<'a>;
}

pub type Foo = impl for<'a> Trait<Assoc<'a> = FooAssoc<'a>>;
pub type FooAssoc<'a> = impl Sized;

struct Struct;
impl Trait for Struct {
    type Assoc<'a> = &'a u32;
}

#[define_opaque(Foo)]
fn foo() -> Foo {
    Struct
    //~^ ERROR: expected generic lifetime parameter, found `'a`
}

fn main() {}
