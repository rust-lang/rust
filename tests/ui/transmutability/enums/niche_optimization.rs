//@ check-pass
//! Checks that niche optimizations are encoded correctly.
#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: false,
                lifetimes: false,
                safety: true,
                validity: false,
            }
        }>
    {}

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: false,
                lifetimes: false,
                safety: true,
                validity: true,
            }
        }>
    {}
}

#[repr(u8)] enum V0 { V = 0 }
#[repr(u8)] enum V1 { V = 1 }
#[repr(u8)] enum V2 { V = 2 }
#[repr(u8)] enum V253 { V = 253 }
#[repr(u8)] enum V254 { V = 254 }
#[repr(u8)] enum V255 { V = 255 }

fn bool() {
    enum OptionLike {
        A(bool),
        B,
    }

    const _: () = {
        assert!(std::mem::size_of::<OptionLike>() == 1);
    };

    assert::is_transmutable::<OptionLike, u8>();

    assert::is_transmutable::<bool, OptionLike>();
    assert::is_transmutable::<V0, OptionLike>();
    assert::is_transmutable::<V1, OptionLike>();
    assert::is_transmutable::<V2, OptionLike>();
}

fn one_niche() {
    #[repr(u8)]
    enum N1 {
        S = 0,
        E = 255 - 1,
    }

    enum OptionLike {
        A(N1),
        B,
    }

    const _: () = {
        assert!(std::mem::size_of::<OptionLike>() == 1);
    };

    assert::is_transmutable::<OptionLike, u8>();
    assert::is_transmutable::<V0, OptionLike>();
    assert::is_transmutable::<V1, OptionLike>();
    assert::is_transmutable::<V254, OptionLike>();
}

fn one_niche_alt() {
    #[repr(u8)]
    enum N1 {
        S = 1,
        E = 255 - 1,
    }

    enum OptionLike {
        A(N1),
        B,
        C,
    }

    const _: () = {
        assert!(std::mem::size_of::<OptionLike>() == 1);
    };

    assert::is_transmutable::<OptionLike, u8>();
    assert::is_transmutable::<V1, OptionLike>();
    assert::is_transmutable::<V2, OptionLike>();
    assert::is_transmutable::<V254, OptionLike>();
}

fn two_niche() {
    #[repr(u8)]
    enum Niche {
        S = 0,
        E = 255 - 2,
    }

    enum OptionLike {
        A(Niche),
        B,
        C,
    }

    const _: () = {
        assert!(std::mem::size_of::<OptionLike>() == 1);
    };

    assert::is_transmutable::<OptionLike, u8>();
    assert::is_transmutable::<V0, OptionLike>();
    assert::is_transmutable::<V1, OptionLike>();
    assert::is_transmutable::<V2, OptionLike>();
    assert::is_transmutable::<V253, OptionLike>();
}

fn no_niche() {
    use std::mem::MaybeUninit;

    #[repr(u8)]
    enum Niche {
        S = 0,
        E = 255 - 1,
    }

    enum OptionLike {
        A(Niche),
        B,
        C,
    }

    const _: () = {
        assert!(std::mem::size_of::<OptionLike>() == 1);
    };

    #[repr(C)]
    struct Pair<T, U>(T, U);

    assert::is_transmutable::<V0, Niche>();
    assert::is_transmutable::<V254, Niche>();
    assert::is_transmutable::<Pair<V0, Niche>, OptionLike>();
    assert::is_transmutable::<Pair<V1, MaybeUninit<u8>>, OptionLike>();
    assert::is_transmutable::<Pair<V2, MaybeUninit<u8>>, OptionLike>();
}

fn niche_fields() {
    enum Kind {
        A(bool, bool),
        B(bool),
    }

    assert::is_maybe_transmutable::<u16, Kind>();
}
