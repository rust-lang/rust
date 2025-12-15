//@ edition:2018
#![crate_type = "lib"]

macro_rules! not_pub {
    () => {}
}

pub use not_pub; //~ ERROR: `not_pub` is only public within the crate, and cannot be re-exported outside [E0364]
