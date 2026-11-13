//@ check-fail

//! Reject extensions behind references.

#![crate_type = "lib"]
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<
            Src,
            {
                Assume {
                    alignment: true,
                    lifetimes: true,
                    safety: true,
                    validity: true,
                }
            },
        >,
    {
    }
}

#[repr(C, packed)]
struct Packed<T>(T);

fn reject_extension() {
    #[repr(C, align(2))]
    struct Two(u8);

    #[repr(C, align(4))]
    struct Four(u8);

    // These two types differ in the number of trailing padding bytes they have.
    type Src = Packed<Two>;
    type Dst = Packed<Four>;

    const _: () = {
        use std::mem::size_of;
        assert!(size_of::<Src>() == 2);
        assert!(size_of::<Dst>() == 4);
    };

    assert::is_transmutable::<&Src, &Dst>(); //~ ERROR cannot be safely transmuted
}
