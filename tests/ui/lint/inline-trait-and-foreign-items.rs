#![feature(extern_types)]
#![feature(impl_trait_in_assoc_type)]

#![warn(unused_attributes)]

trait Trait {
    #[inline] //~ WARN attribute cannot be used on
//~| WARN previously accepted
    const X: u32;

    #[inline] //~ ERROR attribute cannot be used on
    type T;

    type U;
}

impl Trait for () {
    #[inline] //~ WARN attribute cannot be used on
//~| WARN previously accepted
    const X: u32 = 0;

    #[inline] //~ ERROR attribute cannot be used on
    type T = Self;

    #[inline] //~ ERROR attribute cannot be used on
    type U = impl Trait; //~ ERROR unconstrained opaque type
}

extern "C" {
    #[inline] //~ ERROR attribute cannot be used on
    static X: u32;

    #[inline] //~ ERROR attribute cannot be used on
    type T;
}

fn main() {}
