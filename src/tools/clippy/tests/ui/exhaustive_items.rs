#![deny(clippy::exhaustive_enums, clippy::exhaustive_structs)]
#![allow(unused)]

fn main() {
    // nop
}

pub mod enums {
    pub enum Exhaustive {
        Foo,
        Bar,
        Baz,
        Quux(String),
    }

    /// Some docs
    #[repr(C)]
    pub enum ExhaustiveWithAttrs {
        Foo,
        Bar,
        Baz,
        Quux(String),
    }

    // no warning, already non_exhaustive
    #[non_exhaustive]
    pub enum NonExhaustive {
        Foo,
        Bar,
        Baz,
        Quux(String),
    }

    // no warning, private
    enum ExhaustivePrivate {
        Foo,
        Bar,
        Baz,
        Quux(String),
    }

    // no warning, private
    #[non_exhaustive]
    enum NonExhaustivePrivate {
        Foo,
        Bar,
        Baz,
        Quux(String),
    }
}

pub mod structs {
    pub struct Exhaustive {
        pub foo: u8,
        pub bar: String,
    }

    // no warning, already non_exhaustive
    #[non_exhaustive]
    pub struct NonExhaustive {
        pub foo: u8,
        pub bar: String,
    }

    // no warning, private fields
    pub struct ExhaustivePrivateFieldTuple(u8);

    // no warning, private fields
    pub struct ExhaustivePrivateField {
        pub foo: u8,
        bar: String,
    }

    // no warning, private
    struct ExhaustivePrivate {
        pub foo: u8,
        pub bar: String,
    }

    // no warning, private
    #[non_exhaustive]
    struct NonExhaustivePrivate {
        pub foo: u8,
        pub bar: String,
    }
}
