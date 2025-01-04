//@ edition: 2024
//@ revisions: classic structural
//@[classic] run-pass
//! Tests for errors from binding with `ref x` under a by-ref default binding mode. These can't be
//! in the same body as tests for other errors, since they're emitted during THIR construction.
#![allow(incomplete_features)]
#![cfg_attr(classic, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    let [&ref x] = &[&0];
    //[structural]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic)] let _: &&u32 = x;

    let [&ref x] = &[&mut 0];
    //[structural]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic)] let _: &&mut u32 = x;

    let [&ref x] = &mut [&0];
    //[structural]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic)] let _: &&u32 = x;

    let [&ref x] = &mut [&mut 0];
    //[structural]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic)] let _: &&mut u32 = x;

    let [&mut ref x] = &mut [&mut 0];
    //[structural]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic)] let _: &&mut u32 = x;

    let [&mut ref mut x] = &mut [&mut 0];
    //[structural]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic)] let _: &mut &mut u32 = x;
}
