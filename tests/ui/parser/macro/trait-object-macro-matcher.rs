// A single lifetime is not parsed as a type.
// `ty` matcher in particular doesn't accept a single lifetime

//@ revisions: e2015 e2021
//@[e2015] edition: 2015
//@[e2021] edition: 2021

macro_rules! m {
    ($t: ty) => {
        let _: $t;
    };
}

fn main() {
    //[e2021]~vv ERROR expected type, found lifetime
    //[e2021]~v ERROR expected type, found lifetime
    m!('static);
    //[e2015]~^ ERROR lifetimes must be followed by `+` to form a trait object type
    //[e2015]~| ERROR lifetimes must be followed by `+` to form a trait object type
    //[e2015]~| ERROR at least one trait is required for an object type
}
