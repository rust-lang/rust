//@ revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(generic_const_exprs))]

struct Bug<S: ?Sized> {
    A: [(); {
        //[full]~^ ERROR overly complex generic constant
        let x: Option<Box<Self>> = None;
        //[min]~^ ERROR generic `Self` types are currently not permitted in anonymous constants
        0
    }],
    B: S
}

fn main() {}
