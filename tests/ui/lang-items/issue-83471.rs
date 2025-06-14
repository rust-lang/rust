// Regression test for the ICE reported in issue #83471.

#![crate_type="lib"]
#![feature(no_core)]
#![no_core]

#[lang = "pointee_sized"]
//~^ ERROR: lang items are subject to change [E0658]
pub trait PointeeSized {}

#[lang = "meta_sized"]
//~^ ERROR: lang items are subject to change [E0658]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
//~^ ERROR: lang items are subject to change [E0658]
trait Sized: MetaSized {}

#[lang = "fn"]
//~^ ERROR: lang items are subject to change [E0658]
//~| ERROR: `fn` lang item must be applied to a trait with 1 generic argument
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
