//@ revisions: enabled disabled
//@[enabled] run-rustfix
#![allow(private_interfaces, dead_code)]
#![cfg_attr(enabled, feature(default_field_values))]
use m::S;

mod m {
    pub struct S {
        pub field: () = (),
        //[disabled]~^ ERROR default values on fields are experimental
        pub field1: Priv = Priv,
        //[disabled]~^ ERROR default values on fields are experimental
        pub field2: Priv = Priv,
        //[disabled]~^ ERROR default values on fields are experimental
    }
    struct Priv;
}

fn main() {
    let _ = S { .. }; // ok
    //[disabled]~^ ERROR base expression required after `..`
    let _ = S { field: (), .. }; // ok
    //[disabled]~^ ERROR base expression required after `..`
    let _ = S { };
    //~^ ERROR missing fields `field`, `field1` and `field2`
    let _ = S { field: () };
    //~^ ERROR missing fields `field1` and `field2`
}
