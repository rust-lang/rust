// check-pass
//! If visibility is assumed, a transmutation should be accepted even if the
//! destination type contains a private field.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_transmutable<Src, Dst, Context>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, { Assume::SAFETY }>
        // safety IS assumed --------------------^^^^^^^^^^^^^^^^^^
    {}
}

mod src {
    #[repr(C)] pub(self) struct Zst;

    #[repr(C)] pub(in super) struct Src {
        pub(self) field: Zst,
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
    assert::is_transmutable::<src::Src, dst::Dst, Context>();
}
