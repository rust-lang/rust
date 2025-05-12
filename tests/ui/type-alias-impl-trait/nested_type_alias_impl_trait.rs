#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

pub type Foo = impl Debug;
pub type Foot = impl Debug;

#[define_opaque(Foo)]
pub fn get_foo() -> Foo {
    5i32
}

#[define_opaque(Foot, Foo)]
pub fn get_foot(_: Foo) -> Foot {
    //~^ ERROR item does not constrain `Foo::{opaque#0}`
    get_foo() //~ ERROR opaque type's hidden type cannot be another opaque type
}

fn main() {
    let _: Foot = get_foot(get_foo());
}
