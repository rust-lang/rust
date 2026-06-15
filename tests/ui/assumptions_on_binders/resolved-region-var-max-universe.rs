//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

// Regression test for an ICE in the `MaxUniverse` region visitor. When computing
// the max universe of a region constraint, a `ReVar` term could already have been
// unified with another region. `universe_of_lt` returns `None` for such a resolved
// variable, so the visitor used to `unwrap()` `None` and panic. It now resolves the
// variable before inspecting its universe.

#![feature(min_generic_const_args, inherent_associated_types, generic_const_items)]

struct Parent<'a> {
    a: &'a str,
}

impl<'a> Parent<'a> {
    type const CT<T: 'a>: usize = 0;
}

fn check/*<T>*/()
where
    [(); Parent::CT::<T>]:,
    //~^ ERROR cannot find type `T` in this scope
{
}

fn main() {}
