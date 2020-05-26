#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

fn main() {
    const C: impl Copy = 0;
    match C {
        C | _ => {} //~ ERROR: opaque types cannot be used in patterns
    }
}
