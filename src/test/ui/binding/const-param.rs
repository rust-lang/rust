// Identifier pattern referring to a const generic parameter is an error (issue #68853).
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn check<const N: usize>() {
    match 1 {
        N => {} //~ ERROR const parameters cannot be referenced in patterns
        _ => {}
    }
}

fn main() {}
