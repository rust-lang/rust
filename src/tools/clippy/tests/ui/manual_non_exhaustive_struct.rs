#![warn(clippy::manual_non_exhaustive)]
#![allow(unused)]
//@no-rustfix
mod structs {
    struct S {
        //~^ ERROR: this seems like a manual implementation of the non-exhaustive pattern
        pub a: i32,
        pub b: i32,
        _c: (),
    }

    // user forgot to remove the private field
    #[non_exhaustive]
    struct Sp {
        //~^ ERROR: this seems like a manual implementation of the non-exhaustive pattern
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
    //~^ ERROR: this seems like a manual implementation of the non-exhaustive pattern

    // user forgot to remove the private field
    #[non_exhaustive]
    struct Tp(pub i32, pub i32, ());
    //~^ ERROR: this seems like a manual implementation of the non-exhaustive pattern

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
