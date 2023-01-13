// This is currently not possible to use projections as union field's type.

#![crate_type = "lib"]

pub trait Identity {
    type Identity;
}

impl<T> Identity for T {
    type Identity = Self;
}

pub type Foo = u8;

pub union Bar {
    a:  <Foo as Identity>::Identity, //~ ERROR
    b: u8,
}
