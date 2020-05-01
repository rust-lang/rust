#![warn(clippy::manual_non_exhaustive)]
#![allow(unused)]

mod enums {
    enum E {
        A,
        B,
        #[doc(hidden)]
        _C,
    }

    // user forgot to remove the marker
    #[non_exhaustive]
    enum Ep {
        A,
        B,
        #[doc(hidden)]
        _C,
    }

    // marker variant does not have doc hidden attribute, should be ignored
    enum NoDocHidden {
        A,
        B,
        _C,
    }

    // name of variant with doc hidden does not start with underscore, should be ignored
    enum NoUnderscore {
        A,
        B,
        #[doc(hidden)]
        C,
    }

    // variant with doc hidden is not unit, should be ignored
    enum NotUnit {
        A,
        B,
        #[doc(hidden)]
        _C(bool),
    }

    // variant with doc hidden is the only one, should be ignored
    enum OnlyMarker {
        #[doc(hidden)]
        _A,
    }

    // variant with multiple markers, should be ignored
    enum MultipleMarkers {
        A,
        #[doc(hidden)]
        _B,
        #[doc(hidden)]
        _C,
    }

    // already non_exhaustive and no markers, should be ignored
    #[non_exhaustive]
    enum NonExhaustive {
        A,
        B,
    }
}

mod structs {
    struct S {
        pub a: i32,
        pub b: i32,
        _c: (),
    }

    // user forgot to remove the private field
    #[non_exhaustive]
    struct Sp {
        pub a: i32,
        pub b: i32,
        _c: (),
    }

    // some other fields are private, should be ignored
    struct PrivateFields {
        a: i32,
        pub b: i32,
        _c: (),
    }

    // private field name does not start with underscore, should be ignored
    struct NoUnderscore {
        pub a: i32,
        pub b: i32,
        c: (),
    }

    // private field is not unit type, should be ignored
    struct NotUnit {
        pub a: i32,
        pub b: i32,
        _c: i32,
    }

    // private field is the only field, should be ignored
    struct OnlyMarker {
        _a: (),
    }

    // already non exhaustive and no private fields, should be ignored
    #[non_exhaustive]
    struct NonExhaustive {
        pub a: i32,
        pub b: i32,
    }
}

mod tuple_structs {
    struct T(pub i32, pub i32, ());

    // user forgot to remove the private field
    #[non_exhaustive]
    struct Tp(pub i32, pub i32, ());

    // some other fields are private, should be ignored
    struct PrivateFields(pub i32, i32, ());

    // private field is not unit type, should be ignored
    struct NotUnit(pub i32, pub i32, i32);

    // private field is the only field, should be ignored
    struct OnlyMarker(());

    // already non exhaustive and no private fields, should be ignored
    #[non_exhaustive]
    struct NonExhaustive(pub i32, pub i32);
}

fn main() {}
