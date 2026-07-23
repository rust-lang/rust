#![warn(clippy::std_instead_of_core)]
#![warn(clippy::std_instead_of_alloc)]
#![allow(unused_imports)]
//@no-rustfix

#[rustfmt::skip]
fn issue14982() {
    // FIXME(oli-obk): make this report again
    use std::{collections::HashMap, hash::Hash};
}

#[rustfmt::skip]
fn issue15143() {
    // FIXME(oli-obk): make this report again per violation
    use std::{error::Error, vec::Vec, fs::File};
    //~^ std_instead_of_core
}

#[rustfmt::skip]
fn pr16964() {
    use std::{
        //~^ std_instead_of_alloc
        // FIXME(oli-obk): make this report again
        borrow::Cow,
        collections::BTreeSet,
        ffi::OsString,
    };
}
