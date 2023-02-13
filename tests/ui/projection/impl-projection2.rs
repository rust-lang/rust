// check-pass

#![crate_type = "lib"]

pub trait Identity {
    type Identity: ?Sized;
}

impl<T: ?Sized> Identity for T {
    type Identity = S;
}

pub struct S;
pub struct Bar;

impl <Bar as Identity>::Identity {
    pub fn foo(&self) {}
}

impl S {
    pub fn bar(&self) {
        self.foo();
    }
}
