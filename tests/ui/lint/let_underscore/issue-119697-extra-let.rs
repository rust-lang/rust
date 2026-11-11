#![deny(let_underscore_drop)]
#![feature(type_alias_impl_trait)]

pub struct Foo {
    /// This type must have nontrivial drop glue
    field: String,
}

pub type Tait = impl Sized;

#[define_opaque(Tait)]
pub fn ice_cold(beverage: Tait) {
    // Must destructure at least one field of `Foo`
    let Foo { field } = beverage;
    // boom
    _ = field; //~ ERROR non-binding let on a type that has a destructor

    let _ = field; //~ ERROR non-binding let on a type that has a destructor
}

pub fn main() {}
