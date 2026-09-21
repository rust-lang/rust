//@ aux-build: incoherent_impl2.rs
//@ build-aux-docs

#![feature(rustc_attrs)]
#![allow(internal_features)]

extern crate incoherent_impl2 as baz;

pub mod error {
    pub use baz::Error;
}

impl baz::Error {
    #[rustc_allow_incoherent_impl]
    pub fn new() -> Self {
        Self
    }
}
