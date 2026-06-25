#![feature(transmutability)]
#![feature(unsafe_fields)]
#![allow(dead_code, incomplete_features)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn nothing<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::NOTHING }>,
    {
    }

    pub fn safety<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>,
    {
    }
}

mod owner {
    #[repr(C)]
    pub struct PublicField {
        pub field: u8,
    }

    #[repr(C)]
    pub struct PrivateField {
        field: u8,
    }

    #[repr(C)]
    pub struct PublicUnsafeField {
        pub unsafe field: u8,
    }
}

fn main() {
    assert::nothing::<u8, owner::PublicField>();
    assert::nothing::<owner::PublicField, u8>();

    assert::nothing::<u8, owner::PrivateField>(); //~ ERROR cannot be safely transmuted
    assert::safety::<u8, owner::PrivateField>();
    assert::nothing::<owner::PrivateField, u8>();

    assert::nothing::<u8, owner::PublicUnsafeField>(); //~ ERROR cannot be safely transmuted
    assert::safety::<u8, owner::PublicUnsafeField>();
    assert::nothing::<owner::PublicUnsafeField, u8>();
}
