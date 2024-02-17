//@ edition:2018
//@ aux-build:anon-params-edition-hygiene.rs

// This warning is still surfaced
#![allow(anonymous_parameters)]

#[macro_use]
extern crate anon_params_edition_hygiene;

generate_trait_2015_ident!(u8);
// FIXME: Edition hygiene doesn't work correctly with `tt`s in this case.
generate_trait_2015_tt!(u8); //~ ERROR expected one of `:`, `@`, or `|`, found `)`

fn main() {}
