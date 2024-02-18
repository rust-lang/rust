//@ revisions: full min
#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

fn func<A, const F: fn(inner: A)>(outer: A) {
    //~^ ERROR: the type of const parameters must not depend on other generic parameters
    F(outer);
}

fn main() {}
