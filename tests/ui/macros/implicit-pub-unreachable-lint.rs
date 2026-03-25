//@ edition:2018
#![crate_type = "lib"]
#![feature(macro_implicit_pub)]
#![deny(unreachable_pub)]

macro_rules! not_reexported {
    () => {}
}

mod inner {
    pub use not_reexported; //~ ERROR: unreachable `pub` item [unreachable_pub]
}
