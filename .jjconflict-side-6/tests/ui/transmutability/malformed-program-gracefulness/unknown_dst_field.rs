// An unknown destination type should be gracefully handled.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(incomplete_features)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src>
    {}
}

fn should_gracefully_handle_unknown_dst_field() {
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst(Missing); //~ ERROR cannot find type
    assert::is_transmutable::<Src, Dst>(); //~ ERROR cannot be safely transmuted
}

fn should_gracefully_handle_unknown_dst_ref_field() {
    #[repr(C)] struct Src(&'static Src);
    #[repr(C)] struct Dst(&'static Missing); //~ ERROR cannot find type
    assert::is_transmutable::<Src, Dst>(); //~ ERROR cannot be safely transmuted
}
