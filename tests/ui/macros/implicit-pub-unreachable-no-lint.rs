//@ check-pass
//@ edition:2018
#![crate_type = "lib"]

macro_rules! not_reexported {
    () => {}
}
