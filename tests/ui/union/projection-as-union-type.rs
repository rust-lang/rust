// Ensures that we can use projections as union field's type.
//@ check-pass

#![crate_type = "lib"]

pub trait Identity {
    type Identity;
}

impl<T> Identity for T {
    type Identity = Self;
}

pub type Foo = u8;

pub union Bar {
    pub a: <Foo as Identity>::Identity,
    pub b: u8,
}
