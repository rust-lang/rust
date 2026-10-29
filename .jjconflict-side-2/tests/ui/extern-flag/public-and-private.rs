//@ aux-crate:priv:somedep=somedep.rs
//@ compile-flags: -Zunstable-options --extern somedep
//@ edition:2018

#![deny(exported_private_dependencies)]

// Having a flag with `priv` and one without, will remain private (it is sticky).

pub struct PublicType {
    pub field: somedep::S, //~ ERROR from private dependency
}

fn main() {}
