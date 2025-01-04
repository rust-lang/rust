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
        //[structural]~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &mut Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(Some((&mut _)))) = &Some(Some(&mut Some(0))) {
        //[structural]~^ ERROR: mismatched types
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
    }
}

// TODO: these should be mutability mismatches on `structural`
fn structural_errors_0() {
    let &[&mut x] = &&mut [0];
    //[structural]~^ ERROR: mismatched types
    let _: u32 = x;
    //[structural]~^ ERROR: mismatched types

    let &[&mut x] = &mut &mut [0];
    //[structural]~^ ERROR: mismatched types
    let _: u32 = x;
    //[structural]~^ ERROR: mismatched types

    let &[&mut ref x] = &&mut [0];
    //[structural]~^ ERROR: mismatched types
    let _: &u32 = x;

    let &[&mut ref x] = &mut &mut [0];
    //[structural]~^ ERROR: mismatched types
    let _: &u32 = x;

    let &[&mut mut x] = &&mut [0];
    //[structural]~^ ERROR: mismatched types
    //[structural]~| ERROR: binding cannot be both mutable and by-reference
    let _: u32 = x;
    //[structural]~^ ERROR: mismatched types

    let &[&mut mut x] = &mut &mut [0];
    //[structural]~^ ERROR: mismatched types
    //[structural]~| ERROR: binding cannot be both mutable and by-reference
    let _: u32 = x;
    //[structural]~^ ERROR: mismatched types
}

fn structural_errors_1() {
    let [&(mut x)] = &[&0];
    //[structural]~^ ERROR: binding cannot be both mutable and by-reference
    let _: &u32 = x;

    let [&(mut x)] = &mut [&0];
    //[structural]~^ ERROR: binding cannot be both mutable and by-reference
    let _: &u32 = x;
}

// TODO: these should be mutability mismatches on `structural`
fn structural_errors_2() {
    let [&&mut x] = &[&mut 0];
    //[structural]~^ ERROR: mismatched types
    let _: u32 = x;
    //[structural]~^ ERROR: mismatched types

    let [&&mut x] = &mut [&mut 0];
    let _: u32 = x;

    let [&&mut ref x] = &[&mut 0];
    //[structural]~^ ERROR: mismatched types
    let _: &u32 = x;

    let [&&mut ref x] = &mut [&mut 0];
    let _: &u32 = x;

    let [&&mut mut x] = &[&mut 0];
    //[structural]~^ ERROR: binding cannot be both mutable and by-reference
    //[structural]~| ERROR: mismatched types
    let _: u32 = x;
    //[structural]~^ ERROR: mismatched types

    let [&&mut mut x] = &mut [&mut 0];
    let _: u32 = x;
}

fn classic_errors_0() {
    let [&mut x] = &[&mut 0];
    //[classic]~^ ERROR: mismatched types
    //[classic]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let [&mut &x] = &[&mut 0];
    //[classic]~^ ERROR: mismatched types
    //[classic]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let [&mut &ref x] = &[&mut 0];
    //[classic]~^ ERROR: mismatched types
    //[classic]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let [&mut &(mut x)] = &[&mut 0];
    //[classic]~^ ERROR: mismatched types
    //[classic]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;
}
