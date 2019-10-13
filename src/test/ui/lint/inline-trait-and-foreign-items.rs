#![feature(extern_types)]
#![feature(type_alias_impl_trait)]

trait Trait {
    #[inline] //~ ERROR attribute should be applied to function or closure
    const X: u32;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T;

    type U;
}

impl Trait for () {
    #[inline] //~ ERROR attribute should be applied to function or closure
    const X: u32 = 0;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T = Self;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type U = impl Trait; //~ ERROR could not find defining uses
}

extern {
    #[inline] //~ ERROR attribute should be applied to function or closure
    static X: u32;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T;
}

fn main() {}
