// Check that we do not allow coercions to object
// unsafe trait objects in match arms

#![feature(dyn_compatible_for_dispatch)]

trait Trait: Sized {}

struct S;

impl Trait for S {}

struct R;

impl Trait for R {}

fn opt() -> Option<()> {
    Some(())
}

fn main() {
    match opt() {
        Some(()) => &S,
        None => &R,  //~ ERROR E0308
    }
    let t: &dyn Trait = match opt() { //~ ERROR E0038
        Some(()) => &S, //~ ERROR E0038
        None => &R,
    };
}
