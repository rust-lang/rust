#![feature(associated_type_defaults)]

pub trait Tr {
    type Assoc = u8;
    type Assoc2 = Self::Assoc;
    const C: u8 = 11;
    fn foo(&self) {}
}

impl Tr for () {
    type Assoc = ();
    //~^ ERROR need to be reimplemented as `Assoc` was overridden: `Assoc2`, `C`, `foo`
}

fn main() {}
