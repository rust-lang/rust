#![feature(type_alias_impl_trait)]

trait Foo {}
impl Foo for () {}
impl Foo for i32 {}

type Bar<T: Foo> = impl std::fmt::Debug;
#[define_opaque(Bar)]
fn defining_use<T: Foo>() -> Bar<T> {
    42
}

trait Bop {}

impl Bop for Bar<()> {}

// If the hidden type is the same, this is effectively a second impl for the same type.
impl Bop for Bar<i32> {}
//~^ ERROR conflicting implementations

type Barr = impl std::fmt::Debug;
#[define_opaque(Barr)]
fn defining_use2() -> Barr {
    42
}

// Even completely different opaque types must conflict.
impl Bop for Barr {}
//~^ ERROR conflicting implementations

// And obviously the hidden type must conflict, too.
impl Bop for i32 {}
//~^ ERROR conflicting implementations

fn main() {}
