//@ compile-flags: --crate-type=lib
#![feature(required_inlining)]

#[inline(required("maintain security properties"))]
pub fn required() {
}

#[inline(must("maintain security properties"))]
pub fn must() {
}
