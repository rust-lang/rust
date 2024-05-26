//@ known-bug: rust-lang/rust#123255
//@ edition:2021
#![crate_type = "lib"]

pub fn a() {}

mod handlers {
    pub struct C(&());
    pub fn c() -> impl Fn() -> C {
        let a1 = ();
        || C((crate::a(), a1).into())
    }
}
