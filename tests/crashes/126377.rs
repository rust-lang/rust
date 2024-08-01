//@ known-bug: rust-lang/rust#126377

#![feature(effects)]
#![feature(generic_const_exprs)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_transmutable<
        Src,
        Dst,
        const ASSUME_ALIGNMENT: bool,
        const ASSUME_LIFETIMES: bool,
        const ASSUME_SAFETY: bool,
        const ASSUME_VALIDITY: bool,
    >()
    where
        Dst: BikeshedIntrinsicFrom<
            Src,
            {  }
        >,
    {}
}

const fn from_options() -> Assume {
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;
    assert::is_transmutable::<Src, Dst, {0u8}, false, false, false>();
}
