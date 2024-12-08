//@ check-pass
//@ aux-crate:noprelude:somedep=somedep.rs
//@ compile-flags: -Zunstable-options --extern somedep
//@ edition:2018

// Having a flag with `noprelude` and one without, will add to the prelude.

fn main() {
    somedep::somefun();
}
