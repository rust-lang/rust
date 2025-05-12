//@ revisions: stable2021 classic2021 structural2021 classic2024 structural2024
//@[stable2021] edition: 2021
//@[classic2021] edition: 2021
//@[structural2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@ dont-require-annotations: NOTE

//! Tests for errors from binding with `ref x` under a by-ref default binding mode in edition 2024.
//! These can't be in the same body as tests for other errors, since they're emitted during THIR
//! construction. The errors on stable edition 2021 Rust are unrelated.
#![allow(incomplete_features)]
#![cfg_attr(any(classic2021, classic2024), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural2021, structural2024), feature(ref_pat_eat_one_layer_2024_structural))]

/// These only fail on the eat-inner variant of the new edition 2024 pattern typing rules.
/// The eat-outer variant eats the inherited reference, so binding with `ref` isn't a problem.
fn errors_from_eating_the_real_reference() {
    let [&ref x] = &[&0];
    //[structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&u32 = x;

    let [&ref x] = &mut [&0];
    //[structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&u32 = x;

    let [&mut ref x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&mut u32 = x;

    let [&mut ref mut x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &mut u32 = x;
    #[cfg(classic2024)] let _: &mut &mut u32 = x;
}

/// To make absolutely sure binding with `ref` ignores inherited references on stable, let's
/// quarantine these typeck errors (from using a `&` pattern to match a `&mut` reference type).
fn errors_from_eating_the_real_reference_caught_in_hir_typeck_on_stable() {
    let [&ref x] = &[&mut 0];
    //[stable2021]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[structural2024]~^^^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(classic2021, structural2021))] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&mut u32 = x;

    let [&ref x] = &mut [&mut 0];
    //[stable2021]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[structural2024]~^^^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(classic2021, structural2021))] let _: &u32 = x;
    #[cfg(classic2024)] let _: &&mut u32 = x;
}

/// This one also needs to be quarantined for a typeck error on `classic2024` (eat-outer).
fn errors_dependent_on_eating_order_caught_in_hir_typeck_when_eating_outer() {
    let [&mut ref x] = &[&mut 0];
    //[classic2024]~^ ERROR: mismatched types
    //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    //[structural2024]~^^^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &u32 = x;
}

/// These should be errors in all editions. In edition 2024, they should be caught by the pattern
/// typing rules disallowing `ref` when there's an inherited reference. In old editions where that
/// resets the binding mode, they're borrowck errors due to binding with `ref mut`.
/// As a quirk of how the edition 2024 error is emitted during THIR construction, it ends up going
/// through borrowck as well, using the old `ref` behavior as a fallback, so we get that error too.
fn borrowck_errors_in_old_editions() {
    let [ref mut x] = &[0];
    //~^ ERROR: cannot borrow data in a `&` reference as mutable
    //[classic2024,structural2024]~| ERROR: binding modifiers may only be written when the default binding mode is `move`
}

/// The remaining tests are purely for testing `ref` bindings in the presence of an inherited
/// reference. These should always fail on edition 2024 and succeed on edition 2021.
pub fn main() {
    let [ref x] = &[0];
    //[classic2024,structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &u32 = x;

    let [ref x] = &mut [0];
    //[classic2024,structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &u32 = x;

    let [ref mut x] = &mut [0];
    //[classic2024,structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(stable2021, classic2021, structural2021))] let _: &mut u32 = x;
}
