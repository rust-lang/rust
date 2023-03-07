// check-pass
//! The presence of a private variant in the source type does not affect
//! transmutability.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, Dst, Context>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context> // safety is NOT assumed
    {}
}

mod src {
    #[derive(Copy, Clone)]
    #[repr(C)] pub(in super) struct Zst;

    #[repr(C)] pub(in super) union Src {
        pub(self) field: Zst, // <- private variant
    }
}

mod dst {
    #[repr(C)] pub(in super) struct Zst;

    #[repr(C)] pub(in super) struct Dst {
        pub(in super) field: Zst,
    }
}

fn test() {
    struct Context;
    assert::is_transmutable::<src::Src, dst::Dst, Context>();
}
