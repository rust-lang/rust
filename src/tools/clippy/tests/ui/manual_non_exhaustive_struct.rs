#![warn(clippy::manual_non_exhaustive)]
#![allow(unused)]
//@no-rustfix
pub mod structs {
    pub struct S {
        //~^ manual_non_exhaustive
        pub a: i32,
        pub b: i32,
        _c: (),
    }

    // user forgot to remove the private field
    #[non_exhaustive]
    pub struct Sp {
        //~^ manual_non_exhaustive
        pub a: i32,
        pub b: i32,
        _c: (),
    }

    // some other fields are private, should be ignored
    pub struct PrivateFields {
        a: i32,
        pub b: i32,
        _c: (),
    }

    pub struct NoUnderscore {
        //~^ manual_non_exhaustive
        pub a: i32,
        pub b: i32,
        c: (),
    }

    // private field is not unit type, should be ignored
    pub struct NotUnit {
        pub a: i32,
        pub b: i32,
        _c: i32,
    }

    // private field is the only field, should be ignored
    pub struct OnlyMarker {
        _a: (),
    }

    // already non exhaustive and no private fields, should be ignored
    #[non_exhaustive]
    pub struct NonExhaustive {
        pub a: i32,
        pub b: i32,
    }
}

pub mod tuple_structs {
    pub struct T(pub i32, pub i32, ());
    //~^ manual_non_exhaustive

    // user forgot to remove the private field
    #[non_exhaustive]
    pub struct Tp(pub i32, pub i32, ());
    //~^ manual_non_exhaustive

    // some other fields are private, should be ignored
    pub struct PrivateFields(pub i32, i32, ());

    // private field is not unit type, should be ignored
    pub struct NotUnit(pub i32, pub i32, i32);

    // private field is the only field, should be ignored
    pub struct OnlyMarker(());

    // already non exhaustive and no private fields, should be ignored
    #[non_exhaustive]
    pub struct NonExhaustive(pub i32, pub i32);
}

mod private {
    // Don't lint structs that are not actually public as `#[non_exhaustive]` only applies to
    // external crates. The manual pattern can still be used to get module local non exhaustiveness
    pub struct NotPublic {
        pub a: i32,
        pub b: i32,
        _c: (),
    }
}
