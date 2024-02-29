// An unknown destination type should be gracefully handled.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src>
    {}
}

fn should_gracefully_handle_unknown_dst_field() {
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst(Missing); //~ cannot find type
    assert::is_transmutable::<Src, Dst>(); //~ ERROR cannot be safely transmuted
}
