#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<
        Src,
        Dst,
        const ASSUME_ALIGNMENT: bool,
        const ASSUME_LIFETIMES: bool,
        const ASSUME_VALIDITY: bool,
        const ASSUME_VISIBILITY: bool,
    >()
    where
        Dst: BikeshedIntrinsicFrom< //~ ERROR trait takes at most 2 generic arguments but 5 generic arguments were supplied
        //~^ ERROR: the constant `ASSUME_ALIGNMENT` is not of type `Assume`
            Src,
            ASSUME_ALIGNMENT, //~ ERROR: mismatched types
            ASSUME_LIFETIMES,
            ASSUME_VALIDITY,
            ASSUME_VISIBILITY,
        >,
    {}
}

fn via_const() {
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    const FALSE: bool = false;

    assert::is_transmutable::<Src, Dst, FALSE, FALSE, FALSE, FALSE>();
}
