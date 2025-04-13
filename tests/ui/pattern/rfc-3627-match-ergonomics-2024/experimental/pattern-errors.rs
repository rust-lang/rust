//@ revisions: stable2021 classic2021 structural2021 classic2024 structural2024
//@[stable2021] edition: 2021
//@[classic2021] edition: 2021
//@[structural2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@ dont-require-annotations: NOTE

//! Test cases for poorly-typed patterns in edition 2024 which are caught by HIR typeck. These must
//! be separate from cases caught by MIR borrowck or the latter errors may not be emitted.
#![allow(incomplete_features)]
#![cfg_attr(any(classic2021, classic2024), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural2021, structural2024), feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(&mut x) = &Some(&mut 0) {
        //[classic2024]~^ ERROR: mismatched types
        //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
        #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
        #[cfg(structural2024)] let _: &u32 = x;
    }
    if let Some(&mut Some(&x)) = &Some(&mut Some(0)) {
        //[stable2021,classic2021,structural2021,classic2024]~^ ERROR: mismatched types
        //[stable2021,classic2021,structural2021]~| NOTE expected integer, found `&_`
        //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
        #[cfg(structural2024)] let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &Some(Some(&mut 0)) {
        //[classic2024]~^ ERROR: mismatched types
        //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
        #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
        #[cfg(structural2024)] let _: &u32 = x;
    }

    if let Some(&mut Some(&_)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
        //[stable2021,classic2021,structural2021]~| NOTE types differ in mutability
        //[classic2024,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    }
    if let Some(&Some(&mut x)) = &Some(&mut Some(0)) {
        //[stable2021,classic2021,structural2021,structural2024]~^ ERROR: mismatched types
        //[stable2021]~| NOTE types differ in mutability
        //[classic2021,structural2021]~| NOTE expected integer, found `&mut _`
        //[structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
        #[cfg(classic2024)] let _: u32 = x;
    }
    if let Some(&Some(&mut _)) = &mut Some(&Some(0)) {
        //~^ ERROR: mismatched types
        //[stable2021,classic2021,structural2021]~| NOTE expected integer, found `&mut _`
        //[classic2024,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    }
    if let Some(&Some(Some(&mut x))) = &Some(Some(&mut Some(0))) {
        //[stable2021,structural2021,structural2024]~^ ERROR: mismatched types
        //[stable2021]~| NOTE expected `Option<&mut Option<{integer}>>`, found `&_`
        //[structural2021,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
        #[cfg(any(classic2021, classic2024))] let _: u32 = x;
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
        //[stable2021]~| NOTE expected `Option<{integer}>`, found `&mut _`
        //[classic2021,structural2021,classic2024,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    }
}

fn structural_errors_0() {
    let &[&mut x] = &&mut [0];
    //[stable2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE expected integer, found `&mut _`
    //[structural2021,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(any(classic2021, classic2024))] let _: u32 = x;

    let &[&mut x] = &mut &mut [0];
    //[stable2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[structural2021,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(any(classic2021, classic2024))] let _: u32 = x;

    let &[&mut ref x] = &&mut [0];
    //[stable2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE expected integer, found `&mut _`
    //[structural2021,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(any(classic2021, classic2024))] let _: &u32 = x;

    let &[&mut ref x] = &mut &mut [0];
    //[stable2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[structural2021,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(any(classic2021, classic2024))] let _: &u32 = x;

    let &[&mut mut x] = &&mut [0];
    //[stable2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE expected integer, found `&mut _`
    //[structural2021,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(any(classic2021, classic2024))] let _: u32 = x;

    let &[&mut mut x] = &mut &mut [0];
    //[stable2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[structural2021,structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(any(classic2021, classic2024))] let _: u32 = x;
}

fn structural_errors_1() {
    let [&(mut x)] = &[&0];
    //[structural2024]~^ ERROR: binding cannot be both mutable and by-reference
    #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
    #[cfg(classic2024)] let _: &u32 = x;

    let [&(mut x)] = &mut [&0];
    //[structural2024]~^ ERROR: binding cannot be both mutable and by-reference
    #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
    #[cfg(classic2024)] let _: &u32 = x;
}

fn structural_errors_2() {
    let [&&mut x] = &[&mut 0];
    //[stable2021,classic2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2021,structural2021] NOTE expected integer, found `&mut _`
    //[structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(classic2024)] let _: u32 = x;

    let [&&mut x] = &mut [&mut 0];
    //[stable2021,classic2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2021,structural2021] NOTE expected integer, found `&mut _`
    //[structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(classic2024)] let _: u32 = x;

    let [&&mut ref x] = &[&mut 0];
    //[stable2021,classic2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2021,structural2021] NOTE expected integer, found `&mut _`
    //[structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(classic2024)] let _: &u32 = x;

    let [&&mut ref x] = &mut [&mut 0];
    //[stable2021,classic2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2021,structural2021] NOTE expected integer, found `&mut _`
    //[structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(classic2024)] let _: &u32 = x;

    let [&&mut mut x] = &[&mut 0];
    //[stable2021,classic2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2021,structural2021] NOTE expected integer, found `&mut _`
    //[structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(classic2024)] let _: u32 = x;

    let [&&mut mut x] = &mut [&mut 0];
    //[stable2021,classic2021,structural2021,structural2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2021,structural2021] NOTE expected integer, found `&mut _`
    //[structural2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(classic2024)] let _: u32 = x;
}

fn classic_errors_0() {
    let [&mut x] = &[&mut 0];
    //[classic2024]~^ ERROR: mismatched types
    //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
    #[cfg(structural2024)] let _: &u32 = x;

    let [&mut &x] = &[&mut 0];
    //[stable2021,classic2021,structural2021,classic2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE expected integer, found `&_`
    //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(structural2024)] let _: u32 = x;

    let [&mut &ref x] = &[&mut 0];
    //[stable2021,classic2021,structural2021,classic2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE expected integer, found `&_`
    //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(structural2024)] let _: &u32 = x;

    let [&mut &(mut x)] = &[&mut 0];
    //[stable2021,classic2021,structural2021,classic2024]~^ ERROR: mismatched types
    //[stable2021]~| NOTE expected integer, found `&_`
    //[classic2024]~| NOTE cannot match inherited `&` with `&mut` pattern
    #[cfg(structural2024)] let _: u32 = x;
}
