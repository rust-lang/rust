//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat --crate-type=staticlib
//@ build-pass
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(slice_swap_unchecked)]
#![feature(autodiff)]

pub mod safe;
pub mod unsf;
