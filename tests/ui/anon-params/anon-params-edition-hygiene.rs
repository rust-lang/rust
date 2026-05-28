//@ check-pass
//@ edition:2018
//@ aux-build:anon-params-edition-hygiene.rs

// This warning is still surfaced
#![allow(anonymous_parameters)]

#[macro_use]
extern crate anon_params_edition_hygiene;

generate_trait_2015_ident!(u8);
generate_trait_2015_tt!(u8);

fn main() {}
