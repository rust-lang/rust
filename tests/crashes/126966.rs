//@ known-bug: rust-lang/rust#126966
mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src>,
    {
    }
}

#[repr(u32)]
enum Ox00 {
    V = 0x00,
}

#[repr(C, packed(2))]
enum OxFF {
    V = 0xFF,
}

fn test() {
    union Superset {
        a: Ox00,
        b: OxFF,
    }

    assert::is_transmutable::<Superset, Subset>();
}
