//@ edition: 2024
//@ revisions: classic2024 structural2024
//@[classic2024] run-pass
//! Tests for errors from binding with `ref x` under a by-ref default binding mode. These can't be
//! in the same body as tests for other errors, since they're emitted during THIR construction.
#![allow(incomplete_features)]
#![cfg_attr(classic2024, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural2024, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    let [&ref x] = &[&0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &&u32 = x;

    let [&ref x] = &[&mut 0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &&mut u32 = x;

    let [&ref x] = &mut [&0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &&u32 = x;

    let [&ref x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &&mut u32 = x;

    let [&mut ref x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &&mut u32 = x;

    let [&mut ref mut x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &mut &mut u32 = x;
}
