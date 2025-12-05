//@ revisions: stable2021 classic2021 structural2021 classic2024 structural2024
//@[stable2021] edition: 2021
//@[classic2021] edition: 2021
//@[structural2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@[classic2024] run-rustfix
//@[structural2024] run-rustfix
//! Tests for `&` patterns matched against `&mut` reference types where the inner pattern attempts
//! to bind by mutable reference.
#![allow(incomplete_features)]
#![cfg_attr(any(classic2021, classic2024), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural2021, structural2024), feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(&Some(ref mut x)) = &mut Some(Some(0)) {
        //[stable2021]~^ ERROR: mismatched types
        //[classic2021,structural2021,classic2024,structural2024]~^^ ERROR: cannot borrow as mutable inside an `&` pattern
        let _: &mut u8 = x;
    }

    if let &Some(Some(ref mut x)) = &mut Some(Some(0)) {
        //[stable2021]~^ ERROR: mismatched types
        //[classic2021,structural2021,classic2024,structural2024]~^^ ERROR: cannot borrow as mutable inside an `&` pattern
        let _: &mut u8 = x;
    }

    macro_rules! pat {
        ($var:ident) => { ref mut $var };
    }
    let &pat!(x) = &mut 0;
    //[stable2021]~^ ERROR: mismatched types
    //[classic2021,structural2021,classic2024,structural2024]~^^ ERROR: cannot borrow as mutable inside an `&` pattern
    let _: &mut u8 = x;

    let &(ref mut a, ref mut b) = &mut (true, false);
    //[stable2021]~^ ERROR: mismatched types
    //[classic2021,structural2021,classic2024,structural2024]~^^ ERROR: cannot borrow as mutable inside an `&` pattern
    //[classic2021,structural2021,classic2024,structural2024]~| ERROR: cannot borrow as mutable inside an `&` pattern
    let _: &mut bool = a;
    let _: &mut bool = b;

    let &[x] = &mut &mut [0];
    //[stable2021]~^ ERROR: mismatched types
    //[classic2021,classic2024]~^^ ERROR: cannot borrow as mutable inside an `&` pattern
    let _: &u32 = x;
}
