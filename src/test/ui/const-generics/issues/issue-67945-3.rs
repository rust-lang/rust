// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]

struct Bug<S: ?Sized> {
    A: [(); {
        //[full]~^ ERROR constant expression depends on a generic parameter
        let x: Option<Box<Self>> = None;
        //[min]~^ ERROR generic `Self` types are currently not permitted in anonymous constants
        0
    }],
    B: S
}

fn main() {}
