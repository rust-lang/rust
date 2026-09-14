// Regression test for <https://github.com/rust-lang/rust/issues/128525>.
// Using a `generic_const_exprs` API from a dependency in a crate that does not
// enable the feature itself used to ICE ("called `Option::unwrap()` on a `None`
// value" in `ty/sty.rs`) instead of compiling.
//@ aux-build: dep-with-gce-128525.rs
//@ build-pass

extern crate dep_with_gce_128525 as library;

use library::*;

fn main() {
    let mut inner = ImplementsTraitOverConstGeneric::<4>;
    inner.configure(&Config { config: [0, 0, 0, 0] });
}
