#![allow(incomplete_features)]
#![feature(generic_associated_types)]

// FIXME(generic-associated-types) Investigate why this doesn't compile.

trait Iterator {
    //~^ ERROR the requirement `for<'a> <Self as Iterator>::Item<'a>: 'a` is not satisfied
    type Item<'a>: 'a;
}

fn main() {}
