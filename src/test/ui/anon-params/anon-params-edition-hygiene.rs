// check-pass
// edition:2018
// aux-build:anon-params-edition-hygiene.rs

#[macro_use]
extern crate anon_params_edition_hygiene;

generate_trait_2015!(u8);

fn main() {}
