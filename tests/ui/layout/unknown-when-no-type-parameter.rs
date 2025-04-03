#![feature(trivial_bounds)]

//@ error-pattern: the type `<() as Project>::Assoc` has an unknown layout

trait Project {
    type Assoc;
}

fn foo() where (): Project {
    [(); size_of::<<() as Project>::Assoc>()]; //~ ERROR evaluation of constant value failed
}

fn main() {}
