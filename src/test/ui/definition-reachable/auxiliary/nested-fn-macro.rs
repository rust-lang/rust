#![feature(decl_macro)]

mod n {
    pub(crate) mod p {
        pub fn f() -> i32 { 12 }
    }
}

pub macro m() {
    n::p::f()
}
