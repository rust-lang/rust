//! regression test for #148627

#![feature(associated_type_defaults)]

trait Trait {
    type Assoc2
        = ()
    where
        for<const C: usize> [(); C]: Copy;
    //~^ ERROR: only lifetime parameters can be used in this context
}

fn main() {}
