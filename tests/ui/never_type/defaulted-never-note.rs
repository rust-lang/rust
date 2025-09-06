//@ revisions: nofallback fallback

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
//[fallback]~^ note: required by this bound in `foo`
//[fallback]~| note: required by a bound in `foo`
fn smeg() {
    //[nofallback]~^ error: this function depends on never type fallback being `()`
    //[nofallback]~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in Rust 2024 and in a future release in all editions!
    let _x = return;
    foo(_x);
    //[fallback]~^ error: the trait bound
    //[fallback]~| note: the trait `ImplementedForUnitButNotNever` is not implemented
    //[fallback]~| help: trait `ImplementedForUnitButNotNever` is implemented for `()`
    //[fallback]~| note: this error might have been caused
    //[fallback]~| note: required by a bound introduced by this call
    //[fallback]~| help: you might have intended to use the type `()`
}

fn main() {
    smeg();
}
