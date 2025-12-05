//@no-rustfix

#![warn(clippy::std_instead_of_core)]
#![warn(clippy::std_instead_of_alloc)]
#![allow(unused_imports)]

#[rustfmt::skip]
fn issue14982() {
    use std::{collections::HashMap, hash::Hash};
    //~^ std_instead_of_core
}

#[rustfmt::skip]
fn issue15143() {
    use std::{error::Error, vec::Vec, fs::File};
    //~^ std_instead_of_core
    //~| std_instead_of_alloc
}
