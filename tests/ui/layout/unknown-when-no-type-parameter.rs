#![feature(trivial_bounds)]

trait Project {
    type Assoc;
}

fn foo()
where
    (): Project,
{
    [(); size_of::<<() as Project>::Assoc>()];
    //~^ ERROR the type `<() as Project>::Assoc` has an unknown layout
    //~| NOTE inside `std::mem::size_of::<<() as Project>::Assoc>`
    //~| NOTE evaluation of constant value failed
}

fn main() {}
