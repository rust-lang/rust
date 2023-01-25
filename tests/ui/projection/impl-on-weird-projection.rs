//~ ERROR
#![crate_type = "lib"]

pub trait Identity {
    type Identity: ?Sized;
}

impl<T: ?Sized> Identity for T {
    type Identity = Self;
}

pub struct I8<const F: i8>;

impl <I8<{i8::MIN}> as Identity>::Identity {
    pub fn foo(&self) {}
}
