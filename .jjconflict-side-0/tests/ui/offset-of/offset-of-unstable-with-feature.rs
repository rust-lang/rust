//@ check-pass
//@ aux-build:offset-of-staged-api.rs

#![feature(unstable_test_feature)]

use std::mem::offset_of;

extern crate offset_of_staged_api;

use offset_of_staged_api::*;

fn main() {
    offset_of!(Unstable, unstable);
    offset_of!(Stable, stable);
    offset_of!(StableWithUnstableField, unstable);
    offset_of!(StableWithUnstableFieldType, stable);
    offset_of!(StableWithUnstableFieldType, stable.unstable);
    offset_of!(UnstableWithStableFieldType, unstable);
    offset_of!(UnstableWithStableFieldType, unstable.stable);
}
