//@ known-bug: #138266
//@compile-flags: --crate-type=lib
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
pub fn f(mut x: [u8; Box::b]) {
    x[72] = 1;
}
