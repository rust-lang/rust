//@ aux-build:struct_field_default.rs
#![feature(default_field_values)]

extern crate struct_field_default as xc;

use m::S;

mod m {
   pub struct S {
       pub field: () = (),
       pub field1: Priv1 = Priv1 {},
       pub field2: Priv2 = Priv2,
   }
   struct Priv1 {}
   struct Priv2;
}

fn main() {
    let _ = S { field: (), field1: m::Priv1 {} };
    //~^ ERROR missing field `field2`
    //~| ERROR struct `Priv1` is private
    let _ = S { field: (), field1: m::Priv1 {}, field2: m::Priv2 };
    //~^ ERROR struct `Priv1` is private
    //~| ERROR unit struct `Priv2` is private
    let _ = xc::B { a: xc::Priv };
    //~^ ERROR unit struct `Priv` is private
    let _ = xc::C { a: xc::Priv };
    //~^ ERROR unit struct `Priv` is private
}
