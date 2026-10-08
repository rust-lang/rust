//@ edition: 2018
//@ proc-macro: overcapture-attr-pm.rs

#![allow(unused)]
#![deny(impl_trait_overcaptures)]

#[overcapture_attr_pm::rpc]
pub fn from_attr_macro() {}
//~^^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

pub fn plain(x: &u8) -> impl Sized { *x }
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

macro_rules! mk {
    ($ret:ty) => {
        pub fn from_macro_rules(x: &u8) -> $ret { *x }
    };
}

mk!(impl Sized);
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

macro_rules! mk_body {
    () => {
        pub fn from_macro_body(x: &u8) -> impl Sized { *x }
        //~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
        //~| WARN this changes meaning in Rust 2024
    };
}

mk_body!();

fn main() {}
