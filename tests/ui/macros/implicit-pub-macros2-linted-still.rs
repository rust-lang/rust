//@ edition:2018
#![feature(decl_macro)]
#![crate_type = "lib"]
#![deny(unreachable_pub)]

mod inner {
    pub macro m($inner_str:expr) { //~ ERROR: unreachable `pub` item [unreachable_pub]
        struct S;
    }
}
