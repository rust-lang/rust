#![crate_type = "lib"]

mod assert {
    use std::mem::{Assume, TransmuteFrom};
    //~^ ERROR: use of unstable library feature `transmutability`
    //~| ERROR: use of unstable library feature `transmutability`

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src>,
        //~^ ERROR: use of unstable library feature `transmutability`
        //~^^ ERROR: use of unstable library feature `transmutability`
    {
    }
}

#[repr(u32)]
enum Ox00 {
    V = 0x00,
}

#[repr(C, packed(2))]
//~^ ERROR: attribute should be applied to a struct
enum OxFF {
    V = 0xFF,
}

fn test() {
    union Superset {
        a: Ox00,
        //~^ ERROR: field must implement `Copy`
        b: OxFF,
    }

    assert::is_transmutable::<Superset, Subset>();
    //~^ ERROR: cannot find type `Subset`
}
