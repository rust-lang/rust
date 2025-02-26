//@ revisions: classic2021 structural2021 classic2024 structural2024
//@[classic2021] edition: 2021
//@[structural2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@ aux-build:mixed-editions-macros.rs
//! Tests for typing mixed-edition patterns under the `ref_pat_eat_one_layer_2024` and
//! `ref_pat_eat_one_layer_2024_structural` feature gates.
//! This is meant both to check that patterns are typed with edition-appropriate typing rules and
//! that we keep our internal state consistent when mixing editions.
#![allow(incomplete_features, unused)]
#![cfg_attr(any(classic2021, classic2024), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural2021, structural2024), feature(ref_pat_eat_one_layer_2024_structural))]

extern crate mixed_editions_macros;
use mixed_editions_macros::*;

// Tests type equality in a way that avoids coercing `&&T` to `&T`.
trait Eq<T> {}
impl<T> Eq<T> for T {}
fn assert_type_eq<T, U: Eq<T>>(_: T, _: U) {}

/// Make sure binding with `ref` in the presence of an inherited reference is forbidden when and
/// only when the binding is from edition 2024.
fn ref_binding_tests() {
    let match_ctor!(ref x) = &[0];
    //[classic2024,structural2024]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(classic2021, structural2021))] assert_type_eq(x, &0u32);

    let [bind_ref!(y)] = &[0];
    //[classic2021,structural2021]~^ ERROR: binding modifiers may only be written when the default binding mode is `move`
    #[cfg(any(classic2024, structural2024))] assert_type_eq(y, &0u32);
}

/// Likewise, when binding with `mut`.
fn mut_binding_tests() {
    let match_ctor!(mut x) = &[0];
    //[classic2024,structural2024]~^ ERROR: binding cannot be both mutable and by-reference
    #[cfg(any(classic2021, structural2021))] assert_type_eq(x, 0u32);

    let [bind_mut!(y)] = &[0];
    //[classic2021,structural2021]~^ ERROR: binding cannot be both mutable and by-reference
    #[cfg(any(classic2024, structural2024))] assert_type_eq(y, 0u32);
}

/// Make sure reference patterns correspond to one deref on edition 2024 and two on edition 2021.
fn layers_eaten_tests() {
    let match_ctor!(&x) = &[&0];
    #[cfg(any(classic2021, structural2021))] assert_type_eq(x, 0u32);
    #[cfg(any(classic2024, structural2024))] assert_type_eq(x, &0u32);

    let [match_ref!(y)] = &[&0];
    #[cfg(any(classic2021, structural2021))] assert_type_eq(y, &0u32);
    #[cfg(any(classic2024, structural2024))] assert_type_eq(y, 0u32);
}

/// Make sure downgrading mutable binding modes inside shared refs ("Rule 3") doesn't break.
/// This only applies to `ref_pat_eat_one_layer_2024_structural`, which has Rule 3 in all editions;
/// under `ref_pat_eat_one_layer_2024`, these should be errors.
fn rule_3_tests() {
    let match_ref!([x]) = &&mut [0];
    //[classic2021,classic2024]~^ ERROR: cannot borrow data in a `&` reference as mutable
    #[cfg(any(structural2021, structural2024))] assert_type_eq(x, &0u32);

    let &match_ctor!(y) = &&mut [0];
    //[classic2021,classic2024]~^ ERROR: cannot borrow data in a `&` reference as mutable
    #[cfg(any(structural2021, structural2024))] assert_type_eq(y, &0u32);

    let &[bind!(z)] = &&mut [0];
    //[classic2021,classic2024]~^ ERROR: cannot borrow data in a `&` reference as mutable
    #[cfg(any(structural2021, structural2024))] assert_type_eq(z, &0u32);
}

/// Test that the interaction between Rules 3 and 5 doesn't break.
fn rules_3_and_5_tests() {
    let match_ref!([x]) = &mut &mut [0];
    //[classic2021,classic2024]~^ ERROR: cannot borrow as mutable inside an `&` pattern
    #[cfg(any(structural2021, structural2024))] assert_type_eq(x, &0u32);

    let &match_ctor!(y) = &mut &mut [0];
    //[classic2021,classic2024]~^ ERROR: cannot borrow as mutable inside an `&` pattern
    #[cfg(any(structural2021, structural2024))] assert_type_eq(y, &0u32);

    let &[bind!(z)] = &mut &mut [0];
    //[classic2021,classic2024]~^ ERROR: cannot borrow as mutable inside an `&` pattern
    #[cfg(any(structural2021, structural2024))] assert_type_eq(z, &0u32);
}

/// Make sure matching a lone shared reference with a `&` ("Rule 4") doesn't break.
fn rule_4_tests() {
    let match_ref!([x]) = &[0];
    assert_type_eq(x, 0u32);

    let &match_ctor!(y) = &[0];
    assert_type_eq(y, 0u32);
}

/// Make sure matching a `&mut` reference with a `&` pattern ("Rule 5") doesn't break.
fn rule_5_tests() {
    let match_ref!(x) = &mut 0;
    assert_type_eq(x, 0u32);

    // also test inherited references (assumes rule 4)
    let [match_ref!(y)] = &mut [0];
    assert_type_eq(y, 0u32);
}

/// Make sure binding with `ref mut` is an error within a `&` pattern matching a `&mut` reference.
fn rule_5_mutability_error_tests() {
    let match_ref!(ref mut x) = &mut 0;
    //~^ ERROR: cannot borrow as mutable inside an `&` pattern
    let &bind_ref_mut!(x) = &mut 0;
    //~^ ERROR: cannot borrow as mutable inside an `&` pattern

    // also test inherited references (assumes rule 4)
    let [match_ref!(ref mut x)] = &mut [0];
    //~^ ERROR: cannot borrow as mutable inside an `&` pattern
    let [&bind_ref_mut!(x)] = &mut [0];
    //~^ ERROR: cannot borrow as mutable inside an `&` pattern
}

fn main() {}
