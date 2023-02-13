#![crate_type = "lib"]

pub trait Identity {
    type Identity: ?Sized;
}

impl<T: ?Sized> Identity for T {
    type Identity = ();
}

pub struct S;

impl <S as Identity>::Identity { //~ ERROR
    pub fn foo(&self) {}
}
