// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn func<A, const F: fn(inner: A)>(outer: A) {
    //~^ ERROR: using function pointers as const generic parameters is forbidden
    //~| ERROR: the type of const parameters must not depend on other generic parameters
    F(outer);
}

fn main() {}
