//! Ensure that we check the predicates for well-formedness at the definition site.
#![feature(generic_const_items)]

// NB: Below, we're intentionally not using `$ty:` aka WellFormed($ty) otherwise `$ty` would
// obviously get checked for well-formedness regardless of whether we wfcheck where-clauses or not.

const _: () = ()
where
    Vec<str>: Sized; //~ ERROR the size for values of type `str` cannot be known at compilation time

trait Trait {
    const U: ()
    where
        Vec<str>: Sized;
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
}

impl Trait for () {
    const U: () = ()
    where
        Vec<str>: Sized;
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
}

fn main() {}
