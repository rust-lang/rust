#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<
        Src,
        Dst,
        const ASSUME_ALIGNMENT: bool,
        const ASSUME_LIFETIMES: bool,
        const ASSUME_VALIDITY: bool,
        const ASSUME_VISIBILITY: bool,
    >()
    where
        Dst: TransmuteFrom<
                //~^ ERROR trait takes at most 2 generic arguments but 5 generic arguments were supplied
                Src,
                ASSUME_ALIGNMENT,
                ASSUME_LIFETIMES,
                ASSUME_VALIDITY,
                ASSUME_VISIBILITY,
            >,
    {
    }
}

fn via_const() {
    #[repr(C)]
    struct Src;
    #[repr(C)]
    struct Dst;

    const FALSE: bool = false;

    assert::is_transmutable::<Src, Dst, FALSE, FALSE, FALSE, FALSE>();
}
