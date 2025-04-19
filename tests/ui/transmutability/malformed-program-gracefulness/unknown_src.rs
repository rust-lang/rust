// An unknown source type should be gracefully handled.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src>
    {}
}

fn should_gracefully_handle_unknown_src() {
    #[repr(C)] struct Dst;
    assert::is_transmutable::<Src, Dst>(); //~ ERROR cannot find type
}
