#![feature(trivial_bounds)]

trait Project {
    type Assoc;
}

fn foo()
where
    (): Project,
{
    [(); size_of::<<() as Project>::Assoc>()];
    //~^ ERROR entering unreachable code
    //~| NOTE evaluation of `foo::{constant#0}` failed here
}

fn main() {}
