//@ revisions: stable2021 classic2024 structural2024
//@[stable2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@[classic2024] run-pass
//! Tests for errors from binding with `ref x` under a by-ref default binding mode in edition 2024.
//! These can't be in the same body as tests for other errors, since they're emitted during THIR
//! construction. The errors on stable edition 2021 Rust are unrelated.
#![allow(incomplete_features)]
#![cfg_attr(classic2024, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural2024, feature(ref_pat_eat_one_layer_2024_structural))]

/// These only fail on the eat-inner variant of the new edition 2024 pattern typing rules.
/// The eat-outer variant eats the inherited reference, so binding with `ref` isn't a problem.
fn errors_from_eating_the_real_reference() {
    let [&ref x] = &[&0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(stable2021)] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&u32 = x;

    let [&ref x] = &mut [&0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(stable2021)] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&u32 = x;

    let [&mut ref x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(stable2021)] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&mut u32 = x;

    let [&mut ref mut x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(stable2021)] let _: &mut u32 = x;
    #[cfg(classic2024)] let _: &mut &mut u32 = x;

    errors_from_eating_the_real_reference_caught_in_hir_typeck_on_stable();
}

/// To make absolutely sure binding with `ref` ignores inherited references on stable, let's
/// quarantine these typeck errors (from using a `&` pattern to match a `&mut` reference type).
fn errors_from_eating_the_real_reference_caught_in_hir_typeck_on_stable() {
    let [&ref x] = &[&mut 0];
    //[stable2021]~^ ERROR: mismatched types
    //[stable2021]~| types differ in mutability
    //[structural2024]~^^^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &&mut u32 = x;

    let [&ref x] = &mut [&mut 0];
    //[stable2021]~^ ERROR: mismatched types
    //[stable2021]~| types differ in mutability
    //[structural2024]~^^^ ERROR: this pattern relies on behavior which may change in edition 2024
    //[structural2024]~| cannot override to bind by-reference when that is the implicit default
    #[cfg(classic2024)] let _: &&mut u32 = x;
}

pub fn main() {
    errors_from_eating_the_real_reference();
}
