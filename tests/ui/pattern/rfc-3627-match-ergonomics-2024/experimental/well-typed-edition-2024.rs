//@ revisions: stable2021 classic2021 structural2021 classic2024 structural2024
//@[stable2021] edition: 2021
//@[classic2021] edition: 2021
//@[structural2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@[classic2024] run-pass
//@[structural2024] run-pass
//! Test cases for well-typed patterns in edition 2024. These are in their own file to ensure we
//! pass both HIR typeck and MIR borrowck, as we may skip the latter if grouped with failing tests.
#![allow(incomplete_features, unused_mut)]
#![cfg_attr(any(classic2021, classic2024), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural2021, structural2024), feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    // Tests not using match ergonomics. These should always succeed with the same bindings.
    if let Some(&Some(&mut ref x)) = Some(&Some(&mut 0)) {
        let _: &u32 = x;
    }

    // Tests for differences in how many layers of reference are eaten by reference patterns
    if let Some(Some(&x)) = &Some(Some(&0)) {
        #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
        #[cfg(any(classic2024, structural2024))] let _: &u32 = x;
    }
    if let Some(&Some(x)) = &mut Some(&Some(0)) {
        // This additionally tests that `&` patterns can eat inherited `&mut` refs.
        // This is possible on stable when the real reference being eaten is of a `&` type.
        #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
        #[cfg(any(classic2024, structural2024))] let _: &u32 = x;
    }
    if let Some(Some(&&x)) = &Some(Some(&0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021,classic2021,structural2021]~| expected integer, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x;
    }

    // Tests for eating a lone inherited reference
    if let Some(Some(&x)) = &Some(&Some(0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021,classic2021,structural2021]~| expected integer, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x; // TODO: this should hold on `classic2021` and `structural2021`
    }
    if let Some(&Some(x)) = &Some(Some(0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021,classic2021,structural2021]~| expected `Option<{integer}>`, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x; // TODO: this should hold on `classic2021` and `structural2021`
    }
    if let Some(Some(&mut x)) = &mut Some(&mut Some(0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021,classic2021,structural2021]~| expected integer, found `&mut _`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x; // TODO: this should hold on `classic2021` and `structural2021`
    }

    // Tests for `&` patterns matching real `&mut` reference types
    if let Some(&Some(&x)) = Some(&Some(&mut 0)) {
        //[stable2021]~^ mismatched types
        //[stable2021]~| types differ in mutability
        #[cfg(any(classic2021, structural2021, classic2024, structural2024))] let _: u32 = x;
    }

    // Tests for eating only one layer and also eating a lone inherited reference
    if let Some(&Some(&x)) = &Some(&Some(0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021,classic2021,structural2021]~| expected integer, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x;
    }

    // Tests for `&` matching a lone inherited possibly-`&mut` reference
    if let Some(&Some(Some(&x))) = &Some(Some(&mut Some(0))) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021]~| expected `Option<&mut Option<{integer}>>`, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x; // TODO: this should hold on `classic2021` and `structural2021`
    }
    if let Some(&Some(x)) = &mut Some(Some(0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021]~| expected `Option<{integer}>`, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x; // TODO: this should hold on `classic2021` and `structural2021`
    }

    // Tests eating one layer, eating a lone inherited ref, and `&` eating `&mut` (realness varies)
    if let Some(&Some(&x)) = &Some(&mut Some(0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021]~| types differ in mutability
        //[classic2021,structural2021]~| expected integer, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x;
    }
    if let Some(&Some(&x)) = &mut Some(&Some(0)) {
        //[stable2021,classic2021,structural2021]~^ mismatched types
        //[stable2021,classic2021,structural2021]~| expected integer, found `&_`
        #[cfg(any(classic2024, structural2024))] let _: u32 = x;
    }

    // Tests for eat-inner rulesets matching on the outer reference if matching on the inner
    // reference causes a mutability mismatch, i.e. `Deref(EatInner, FallbackToOuter)`:
    let [&mut x] = &mut [&0];
    //[stable2021,classic2021,structural2021]~^ mismatched types
    //[stable2021,classic2021,structural2021]~| types differ in mutability
    // TODO: on `structural2021` `x` should have type `u32`
    #[cfg(any(classic2024, structural2024))] let _: &u32 = x;

    let [&mut ref x] = &mut [&0];
    //[stable2021,classic2021,structural2021]~^ mismatched types
    //[stable2021,classic2021,structural2021]~| types differ in mutability
    // TODO: on `structural2021` `x` should have type `&u32`
    #[cfg(any(classic2024, structural2024))] let _: &&u32 = x;

    let [&mut ref mut x] = &mut [&0];
    //[stable2021,classic2021,structural2021]~^ mismatched types
    //[stable2021,classic2021,structural2021]~| types differ in mutability
    // TODO: this should be a mut borrow behind shared borrow error on `structural2021`
    #[cfg(any(classic2024, structural2024))] let _: &mut &u32 = x;

    let [&mut mut x] = &mut [&0];
    //[stable2021,classic2021,structural2021]~^ mismatched types
    //[stable2021,classic2021,structural2021]~| types differ in mutability
    // TODO: on `structural2021` `x` should have type `u32`
    #[cfg(any(classic2024, structural2024))] let _: &u32 = x;

    let [&mut &x] = &mut [&0];
    //[stable2021,classic2021,structural2021]~^ mismatched types
    //[stable2021,classic2021,structural2021]~| types differ in mutability
    // TODO: [structural2021]~| expected integer, found `&_`
    #[cfg(any(classic2024, structural2024))] let _: u32 = x;

    let [&mut &ref x] = &mut [&0];
    //[stable2021,classic2021,structural2021]~^ mismatched types
    //[stable2021,classic2021,structural2021]~| types differ in mutability
    // TODO: [structural2021]~| expected integer, found `&_`
    #[cfg(any(classic2024, structural2024))] let _: &u32 = x;

    let [&mut &(mut x)] = &mut [&0];
    //[stable2021,classic2021,structural2021]~^ mismatched types
    //[stable2021,classic2021,structural2021]~| types differ in mutability
    // TODO: [structural2021]~| expected integer, found `&_`
    #[cfg(any(classic2024, structural2024))] let _: u32 = x;
}
