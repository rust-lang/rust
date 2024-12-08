#![feature(extern_types)]
#![feature(impl_trait_in_assoc_type)]

#![warn(unused_attributes)]

trait Trait {
    #[inline] //~ WARN `#[inline]` is ignored on constants
    //~^ WARN this was previously accepted
    const X: u32;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T;

    type U;
}

impl Trait for () {
    #[inline] //~ WARN `#[inline]` is ignored on constants
    //~^ WARN this was previously accepted
    const X: u32 = 0;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T = Self;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type U = impl Trait; //~ ERROR unconstrained opaque type
}

extern "C" {
    #[inline] //~ ERROR attribute should be applied to function or closure
    static X: u32;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T;
}

fn main() {}
