//@ edition: 2024
//@ build-pass
//@ compile-flags: -Zprint-type-sizes
//@ needs-deterministic-layouts
//@ revisions: with_feature without_feature
//@ aux-build: has-drop-inside-dep.rs

#![cfg_attr(with_feature, feature(async_drop))]
#![allow(incomplete_features)]

extern crate has_drop_inside_dep;

use has_drop_inside_dep::{with_has_drop, with_has_has_drop};

fn main() {
    with_has_drop();
    with_has_has_drop();
}
