// check-pass

#![crate_type = "lib"]

pub trait Identity {
    type Identity: ?Sized;
}

impl<T: ?Sized> Identity for T {
    type Identity = Self;
}

pub struct S;

impl <S as Identity>::Identity {
    pub fn foo(&self) {}
}

impl S {
    pub fn bar(&self) {
        self.foo();
    }
}
