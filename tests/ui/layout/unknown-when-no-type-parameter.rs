#![feature(trivial_bounds)]

//@ error-pattern: error[E0080]: evaluation of constant value failed
//@ error-pattern: the type `<() as Project>::Assoc` has an unknown layout

trait Project {
    type Assoc;
}

fn foo() where (): Project {
    [(); size_of::<<() as Project>::Assoc>()];
}

fn main() {}
