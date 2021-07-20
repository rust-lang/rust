// Regression test for the ICE reported in issue #83471.

#![crate_type="lib"]
#![feature(no_core)]
#![no_core]

#[lang = "fn"]
//~^ ERROR: language items are subject to change [E0658]
//~| ERROR: `fn` language item must be applied to a trait with 1 generic argument
trait Fn {
    fn call(export_name);
    //~^ ERROR: expected type
    //~| ERROR: `call` function in `fn`/`fn_mut` lang item takes exactly two arguments
    //~| WARNING: anonymous parameters are deprecated
    //~| WARNING: this was previously accepted
}
fn call_through_fn_trait() {
    a()
    //~^ ERROR: cannot find function
}
