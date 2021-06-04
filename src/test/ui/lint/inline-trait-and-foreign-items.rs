#![feature(extern_types)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

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
    type U = impl Trait; //~ ERROR could not find defining uses
}

extern "C" {
    #[inline] //~ ERROR attribute should be applied to function or closure
    static X: u32;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T;
}

fn main() {}
