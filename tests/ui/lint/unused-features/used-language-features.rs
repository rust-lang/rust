//@ check-pass

#![crate_type = "lib"]
#![deny(unused_features)]

// Used language features
#![feature(decl_macro)]
#![cfg_attr(all(), feature(rustc_attrs))]

macro m() {}
pub fn use_decl_macro() {
    m!();
}

#[rustc_dummy]
pub fn use_rustc_attrs() {}
