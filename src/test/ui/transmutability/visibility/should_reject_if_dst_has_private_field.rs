//! Unless visibility is assumed, a transmutation should be rejected if the
//! destination type contains a private field.

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
    #[repr(C)] pub(in super) struct Zst;

    #[repr(C)] pub(in super) struct Src {
        pub(in super) field: Zst,
    }
}

mod dst {
    #[repr(C)] pub(in super) struct Zst;

    #[repr(C)] pub(in super) struct Dst {
        pub(self) field: Zst, // <- private field
    }
}

fn test() {
    struct Context;
    assert::is_transmutable::<src::Src, dst::Dst, Context>(); //~ ERROR cannot be safely transmuted
}
