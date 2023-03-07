#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<
        Src,
        Dst,
        Context,
        const ASSUME_ALIGNMENT: bool,
        const ASSUME_LIFETIMES: bool,
        const ASSUME_VALIDITY: bool,
        const ASSUME_VISIBILITY: bool,
    >()
    where
        Dst: BikeshedIntrinsicFrom< //~ ERROR trait takes at most 3 generic arguments but 6 generic arguments were supplied
            Src,
            Context,
            ASSUME_ALIGNMENT,
            ASSUME_LIFETIMES,
            ASSUME_VALIDITY,
            ASSUME_VISIBILITY,
        >,
    {}
}

fn via_const() {
    struct Context;
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    const FALSE: bool = false;

    assert::is_transmutable::<Src, Dst, Context, FALSE, FALSE, FALSE, FALSE>();
}
