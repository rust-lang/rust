#![crate_type = "lib"]

pub trait Identity {
    const Identity: u32;
}

impl<T: ?Sized> Identity for T {
    const Identity: u32 = 0;
}

pub struct S;

impl <S as Identity>::Identity { //~ ERROR
    pub fn foo(&self) {}
}
