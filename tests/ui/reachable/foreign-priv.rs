// aux-build:foreign-priv-aux.rs
// build-pass

#![crate_type = "lib"]

extern crate foreign_priv_aux;

use foreign_priv_aux::{ImplPrivTrait, PubTrait, Wrapper};

pub fn foo(x: Wrapper<ImplPrivTrait>) {
    x.pub_fn();
}
