// Regression test for the ICE reported in issue #83471.

#![crate_type="lib"]
#![feature(no_core)]
#![no_core]

#[lang = "sized"]
//~^ ERROR: language items are subject to change [E0658]
trait Sized {}

#[lang = "fn"]
//~^ ERROR: language items are subject to change [E0658]
//~| ERROR: `fn` language item must be applied to a trait with 1 generic argument
trait Fn {
    fn call(export_name);
    //~^ ERROR: expected type
    //~| WARNING: anonymous parameters are deprecated
    //~| WARNING: this is accepted in the current edition
}
fn call_through_fn_trait() {
    a()
    //~^ ERROR: cannot find function
}
