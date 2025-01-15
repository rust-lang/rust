//@ edition: 2024
//@ revisions: classic2024 structural2024
//@ run-pass
//! Test cases for well-typed patterns in edition 2024. These are in their own file to ensure we
//! pass both HIR typeck and MIR borrowck, as we may skip the latter if grouped with failing tests.
#![allow(incomplete_features, unused_mut)]
#![cfg_attr(classic2024, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural2024, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(Some(&0)) {
        let _: &u32 = x;
    }
    if let Some(Some(&&x)) = &Some(Some(&0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &Some(Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &mut Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &mut Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &mut Some(&Some(0)) {
        let _: &u32 = x;
    }
    if let Some(&Some(&mut ref x)) = Some(&Some(&mut 0)) {
        let _: &u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(Some(&x))) = &Some(Some(&mut Some(0))) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = Some(&Some(&mut 0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &mut Some(Some(0)) {
        let _: u32 = x;
    }

    // Tests for eat-inner rulesets matching on the outer reference if matching on the inner
    // reference causes a mutability mismatch, i.e. `Deref(EatInner, FallbackToOuter)`:
    let [&mut x] = &mut [&0];
    let _: &u32 = x;

    let [&mut ref x] = &mut [&0];
    let _: &&u32 = x;

    let [&mut ref mut x] = &mut [&0];
    let _: &mut &u32 = x;

    let [&mut mut x] = &mut [&0];
    let _: &u32 = x;

    let [&mut &x] = &mut [&0];
    let _: u32 = x;

    let [&mut &ref x] = &mut [&0];
    let _: &u32 = x;

    let [&mut &(mut x)] = &mut [&0];
    let _: u32 = x;
}
