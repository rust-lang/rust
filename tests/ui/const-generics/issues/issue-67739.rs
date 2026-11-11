//@ revisions: full min
#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

use std::mem;

pub trait Trait {
    type Associated: Sized;

    fn associated_size(&self) -> usize {
        [0u8; mem::size_of::<Self::Associated>()];
        //[min]~^ ERROR constant expression depends on a generic parameter
        //[full]~^^ ERROR unconstrained generic constant
        0
    }
}

fn main() {}
