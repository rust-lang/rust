//@ edition: 2024
//@ revisions: classic structural
//! Test cases for poorly-typed patterns in edition 2024 which are caught by HIR typeck. These must
//! be separate from cases caught by MIR borrowck or the latter errors may not be emitted.
#![allow(incomplete_features)]
#![cfg_attr(classic, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(&mut x) = &mut Some(&0) {
        //[structural]~^ ERROR: mismatched types
        let _: &u32 = x;
    }

    if let Some(&mut x) = &Some(&mut 0) {
        //[classic]~^ ERROR: mismatched types
        let _: &u32 = x;
    }
    if let Some(&mut Some(&x)) = &Some(&mut Some(0)) {
        //[classic]~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &Some(Some(&mut 0)) {
        //[classic]~^ ERROR: mismatched types
        let _: &u32 = x;
    }

    if let Some(&mut Some(&_)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &Some(&mut Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &mut Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(Some((&mut _)))) = &Some(Some(&mut Some(0))) {
        //~^ ERROR: mismatched types
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
    }
}
