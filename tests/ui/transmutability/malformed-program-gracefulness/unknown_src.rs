// An unknown source type should be gracefully handled.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;
    pub struct Context;

    pub fn is_transmutable<Src, Dst, Context>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context>
    {}
}

fn should_gracefully_handle_unknown_src() {
    struct Context;
    #[repr(C)] struct Dst;
    assert::is_transmutable::<Src, Dst, Context>(); //~ cannot find type
}
