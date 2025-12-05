#![feature(generic_const_items)]
#![allow(incomplete_features)]

trait Tr<P> {
    const K: ()
    where
        P: Copy
    where
        P: Eq;
    //~^ ERROR cannot define duplicate `where` clauses on an item
}

// Test that we error on the first where-clause but also that we don't suggest to swap it with the
// body as it would conflict with the second where-clause.
// FIXME(generic_const_items): We should provide a structured sugg to merge the 1st into the 2nd WC.

impl<P> Tr<P> for () {
    const K: ()
    where
        P: Eq
    = ()
    where
        P: Copy;
    //~^^^^^ ERROR where clauses are not allowed before const item bodies
}

fn main() {}
