#![allow(unused)]

mod A {
    pub(crate) type AA = ();
}

mod C {}

mod B {
    use crate::C::{self, AA};
    //~^ ERROR unresolved import `crate::C::AA`
}

fn main() {}
