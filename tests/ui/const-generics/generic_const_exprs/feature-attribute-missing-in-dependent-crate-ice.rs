//! Regression test to ensure that using the `generic_const_exprs` feature in a library crate
//! without enabling it in a dependent crate does not lead to an ICE.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/129882>

//@ aux-build:feature-attribute-missing-in-dependent-crate-ice-aux.rs

extern crate feature_attribute_missing_in_dependent_crate_ice_aux as aux;

struct Wrapper<const F: usize>(i64);

impl<const F: usize> aux::FromSlice for Wrapper<F> {
    fn validate_slice(_: &[[u8; Self::SIZE]]) -> Result<(), aux::Error> {
        //~^ ERROR generic `Self` types are currently not permitted in anonymous constants
        Ok(())
    }
}

fn main() {}
