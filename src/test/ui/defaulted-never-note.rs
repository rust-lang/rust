// We need to opt into the `!` feature in order to trigger the
// requirement that this is testing.
#![feature(never_type)]

#![allow(unused)]

trait Deserialize: Sized {
    fn deserialize() -> Result<Self, String>;
}

impl Deserialize for () {
    fn deserialize() -> Result<(), String> {
        Ok(())
    }
}

trait ImplementedForUnitButNotNever {}

impl ImplementedForUnitButNotNever for () {}

fn foo<T: ImplementedForUnitButNotNever>(_t: T) {}
//~^ NOTE required by `foo`

fn smeg() {
    let _x = return;
    foo(_x);
    //~^ ERROR the trait bound
    //~| NOTE the trait `ImplementedForUnitButNotNever` is not implemented
    //~| NOTE the trait is implemented for `()`
}

fn main() {
    smeg();
}
