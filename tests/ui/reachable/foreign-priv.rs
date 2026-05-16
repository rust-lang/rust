//@ aux-build:foreign-priv-aux.rs
//@ build-pass

#![crate_type = "lib"]
#![allow(unused_unconstructable_pub_structs)]

extern crate foreign_priv_aux;

use foreign_priv_aux::{ImplPrivTrait, PubTrait, Wrapper};

pub fn foo(x: Wrapper<ImplPrivTrait>) {
    x.pub_fn();
}
