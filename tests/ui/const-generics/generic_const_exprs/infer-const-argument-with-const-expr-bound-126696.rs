//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// Used to ICE when the const argument of a call was inferred from the argument
// while the callee had a `[(); N * 2]:` bound.

fn can_double<const N: usize>(x: [(); N])
where
    [(); N * 2]:,
{
    x[0];
    unimplemented!()
}

fn foo<const N: usize>()
where
    [(); (N + 1) * 2]:,
{
    can_double([(); { N + 1 }]);
    // Adding an explicit constant generic causes the ICE to go away
    // can_double::<{N + 1}>([(); { N + 1 }]);
}

fn main() {
    foo::<1>();
}
