//@ edition:2018
#![crate_type = "lib"]
#![deny(unreachable_pub)]

#[macro_export]
macro_rules! not_reexported {
    () => {}
}

mod inner {
    pub use not_reexported; //~ ERROR: unreachable `pub` item [unreachable_pub]
}
