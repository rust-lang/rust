//@ check-pass

// Before RFC 2532, overriding one assoc. type default required overriding all
// provided defaults.

#![feature(associated_type_defaults)]

pub trait Tr {
    type Assoc = u8;
    type Assoc2 = Self::Assoc;
    const C: u8 = 11;
    fn foo(&self) {}
}

impl Tr for () {
    type Assoc = ();
}

fn main() {
    let _: <() as Tr>::Assoc = ();
    let _: <() as Tr>::Assoc2 = ();
}
