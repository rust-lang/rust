#![feature(trivial_bounds)]

trait Project {
    type Assoc;
}

fn foo()
where
    (): Project,
{
    [(); size_of::<<() as Project>::Assoc>()];
    //~^ WARN cannot use constants which depend on trivially-false where clauses
    //~| WARN this was previously accepted by the compiler
    //~| NOTE for more information, see issue #76200
    //~| NOTE `#[warn(const_evaluatable_unchecked)]`
    //~^^^^^ ERROR evaluation of constant value failed
    //~| NOTE the type `<() as Project>::Assoc` has an unknown layout
    //~| NOTE inside `std::mem::size_of::<<() as Project>::Assoc>`
}

fn main() {}
