//! Regression test for <https://github.com/rust-lang/rust/issues/153735>.
#![feature(gca_min_const_items, generic_const_exprs)]

const INC = core::gca!(usize);
//~^ ERROR cannot find value `usize` in this scope
//~| ERROR missing type for `const` item

const ARR: [usize; 1] = [0; core::gca!(INC)];

fn main() {}
