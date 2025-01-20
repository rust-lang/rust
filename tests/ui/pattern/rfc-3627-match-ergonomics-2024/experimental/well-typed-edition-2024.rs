//@ edition: 2024
//@ revisions: classic structural
//@ run-pass
//! Test cases for well-typed patterns in edition 2024. These are in their own file to ensure we
//! pass both HIR typeck and MIR borrowck, as we may skip the latter if grouped with failing tests.
#![allow(incomplete_features)]
#![cfg_attr(classic, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural, feature(ref_pat_eat_one_layer_2024_structural))]

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
    if let &Some(Some(x)) = &Some(&mut Some(0)) {
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
}
