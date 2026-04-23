// Check that the hash of `foo` doesn't change just because we ordered
// the nested items (or even added new ones).

//@ revisions: bpass1 bpass2
//@ compile-flags: -Z query-dep-graph
//@ ignore-backends: gcc
// FIXME(#62277): could be check-pass?

#![crate_type = "rlib"]
#![feature(rustc_attrs)]
#![allow(dead_code)]

#[rustc_clean(except = "owner", cfg = "bpass2")]
pub fn foo() {
    #[cfg(bpass1)]
    pub fn baz() {} // order is different...

    #[rustc_clean(cfg = "bpass2")]
    pub fn bar() {} // but that doesn't matter.

    #[cfg(bpass2)]
    pub fn baz() {} // order is different...

    pub fn bap() {} // neither does adding a new item
}
