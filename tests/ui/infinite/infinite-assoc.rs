//@ aux-build: alias.rs

// issue#128327

extern crate alias;

use alias::Trait;
struct S;
impl Trait for S {
    type T = ();
}
struct A((A, <S as Trait>::T<NOT_EXIST?>));
//~^ ERROR: invalid `?` in type
//~| ERROR: recursive type `A` has infinite size

fn main() {}
