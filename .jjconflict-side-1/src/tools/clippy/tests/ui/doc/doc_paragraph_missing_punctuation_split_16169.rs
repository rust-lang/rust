//@ check-pass
// https://github.com/rust-lang/rust-clippy/issues/16169
#![allow(clippy::mixed_attributes_style)]

///
pub fn dont_warn_inner_outer() {
    //!w
}
///
#[inline]
///w
pub fn dont_warn_split_by_attr() {}
