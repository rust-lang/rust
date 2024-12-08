//@ revisions: e2015 e2024
//@[e2015] edition:2015
//@[e2024] edition:2024

#![warn(missing_fragment_specifier)]

macro_rules! used_arm {
    ( $( any_token $field_rust_type )* ) => {};
    //[e2015]~^ ERROR missing fragment
    //[e2015]~| WARN missing fragment
    //[e2015]~| WARN this was previously accepted
    //[e2024]~^^^^ ERROR missing fragment
    //[e2024]~| ERROR missing fragment
}

macro_rules! used_macro_unused_arm {
    () => {};
    ( $name ) => {};
    //[e2015]~^ WARN missing fragment
    //[e2015]~| WARN this was previously accepted
    //[e2024]~^^^ ERROR missing fragment
}

macro_rules! unused_macro {
    ( $name ) => {};
    //[e2015]~^ WARN missing fragment
    //[e2015]~| WARN this was previously accepted
    //[e2024]~^^^ ERROR missing fragment
}

fn main() {
    used_arm!();
    used_macro_unused_arm!();
}
