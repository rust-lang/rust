//@ edition: 2024
//@ revisions: classic2024 structural2024
//! Test cases for poorly-typed patterns in edition 2024 which are caught by HIR typeck. These must
//! be separate from cases caught by MIR borrowck or the latter errors may not be emitted.
#![allow(incomplete_features)]
#![cfg_attr(classic2024, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural2024, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(&mut x) = &Some(&mut 0) {
        //[classic2024]~^ ERROR: mismatched types
        let _: &u32 = x;
    }
    if let Some(&mut Some(&x)) = &Some(&mut Some(0)) {
        //[classic2024]~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &Some(Some(&mut 0)) {
        //[classic2024]~^ ERROR: mismatched types
        let _: &u32 = x;
    }

    if let Some(&mut Some(&_)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &Some(&mut Some(0)) {
        //[structural2024]~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &mut Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(Some(&mut _))) = &Some(Some(&mut Some(0))) {
        //[structural2024]~^ ERROR: mismatched types
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
    }
}

fn structural_errors_0() {
    let &[&mut x] = &&mut [0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let &[&mut x] = &mut &mut [0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let &[&mut ref x] = &&mut [0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let &[&mut ref x] = &mut &mut [0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let &[&mut mut x] = &&mut [0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let &[&mut mut x] = &mut &mut [0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;
}

fn structural_errors_1() {
    let [&(mut x)] = &[&0];
    //[structural2024]~^ ERROR: binding cannot be both mutable and by-reference
    let _: &u32 = x;

    let [&(mut x)] = &mut [&0];
    //[structural2024]~^ ERROR: binding cannot be both mutable and by-reference
    let _: &u32 = x;
}

fn structural_errors_2() {
    let [&&mut x] = &[&mut 0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let [&&mut x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let [&&mut ref x] = &[&mut 0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let [&&mut ref x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let [&&mut mut x] = &[&mut 0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let [&&mut mut x] = &mut [&mut 0];
    //[structural2024]~^ ERROR: mismatched types
    //[structural2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;
}

fn classic_errors_0() {
    let [&mut x] = &[&mut 0];
    //[classic2024]~^ ERROR: mismatched types
    //[classic2024]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let [&mut &x] = &[&mut 0];
    //[classic2024]~^ ERROR: mismatched types
    //[classic2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;

    let [&mut &ref x] = &[&mut 0];
    //[classic2024]~^ ERROR: mismatched types
    //[classic2024]~| cannot match inherited `&` with `&mut` pattern
    let _: &u32 = x;

    let [&mut &(mut x)] = &[&mut 0];
    //[classic2024]~^ ERROR: mismatched types
    //[classic2024]~| cannot match inherited `&` with `&mut` pattern
    let _: u32 = x;
}
