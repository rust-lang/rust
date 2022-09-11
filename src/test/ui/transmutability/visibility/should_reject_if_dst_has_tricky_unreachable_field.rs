// check-pass
//! NOTE: This test documents a known-bug in the implementation of the
//! transmutability trait. Once fixed, the above "check-pass" header should be
//! removed, and an "ERROR cannot be safely transmuted" annotation should be added at the end
//! of the line starting with `assert::is_transmutable`.
//!
//! Unless visibility is assumed, a transmutation should be rejected if the
//! destination type contains an unreachable field (e.g., a public field with a
//! private type). (This rule is distinct from type privacy, which still may
//! forbid naming such types.)
//!
//! This test exercises a tricky-to-implement instance of this principle: the
//! "pub-in-priv trick". In the below example, the type `dst::private::Zst` is
//! unreachable from `Context`. Consequently, the transmute from `Src` to `Dst`
//! SHOULD be rejected.

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
    mod private {
        #[repr(C)] pub struct Zst; // <- unreachable type
    }

    #[repr(C)] pub(in super) struct Dst {
        pub(in super) field: private::Zst,
    }
}

fn test() {
    struct Context;
    assert::is_transmutable::<src::Src, dst::Dst, Context>();
}
