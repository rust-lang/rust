// revisions: nofallback fallback
//[nofallback] run-pass
//[fallback] check-fail

// We need to opt into the `never_type_fallback` feature
// to trigger the requirement that this is testing.
#![cfg_attr(fallback, feature(never_type, never_type_fallback))]

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
//[fallback]~^ NOTE required by this bound in `foo`
//[fallback]~| NOTE required by a bound in `foo`
fn smeg() {
    let _x = return;
    foo(_x);
    //[fallback]~^ ERROR the trait bound
    //[fallback]~| NOTE the trait `ImplementedForUnitButNotNever` is not implemented
    //[fallback]~| NOTE this trait is implemented for `()`
    //[fallback]~| NOTE this error might have been caused
    //[fallback]~| HELP did you intend
}

fn main() {
    smeg();
}
