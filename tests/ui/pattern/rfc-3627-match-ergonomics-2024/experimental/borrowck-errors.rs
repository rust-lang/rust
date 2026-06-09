//@ revisions: stable2021 classic2021 structural2021 classic2024 structural2024
//@[stable2021] edition: 2021
//@[classic2021] edition: 2021
//@[structural2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@ dont-require-annotations: NOTE

//! Tests for pattern errors not handled by the pattern typing rules, but by borrowck.
#![allow(incomplete_features)]
#![cfg_attr(any(classic2021, classic2024), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural2021, structural2024), feature(ref_pat_eat_one_layer_2024_structural))]

/// These patterns additionally use `&` to match a `&mut` reference type, which causes compilation
/// to fail in HIR typeck on stable. As such, they need to be separate from the other tests.
fn errors_caught_in_hir_typeck_on_stable() {
    let [&x] = &[&mut 0];
    //[stable2021]~^ ERROR mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2024]~^^^ ERROR: cannot move out of type
    #[cfg(any(classic2021, structural2021))] let _: u32 = x;
    #[cfg(structural2024)] let _: &u32 = x;

    let [&x] = &mut [&mut 0];
    //[stable2021]~^ ERROR mismatched types
    //[stable2021]~| NOTE types differ in mutability
    //[classic2024]~^^^ ERROR: cannot move out of type
    #[cfg(any(classic2021, structural2021))] let _: u32 = x;
    #[cfg(structural2024)] let _: &u32 = x;
}

pub fn main() {
    if let Some(&Some(x)) = Some(&Some(&mut 0)) {
        //~^ ERROR: cannot move out of a shared reference [E0507]
        let _: &u32 = x;
    }

    let &ref mut x = &0;
    //~^ ERROR cannot borrow data in a `&` reference as mutable [E0596]

    // For 2021 edition, this is also a regression test for #136223
    // since the maximum mutability is downgraded during the pattern check process.
    if let &Some(Some(x)) = &Some(&mut Some(0)) {
        //[stable2021,classic2021,classic2024]~^ ERROR: cannot borrow data in a `&` reference as mutable
        #[cfg(any(structural2021, structural2024))] let _: &u32 = x;
    }

    let &[x] = &&mut [0];
    //[stable2021,classic2021,classic2024]~^ ERROR: cannot borrow data in a `&` reference as mutable
    #[cfg(any(structural2021, structural2024))] let _: &u32 = x;

    let [&mut x] = &mut [&mut 0];
    //[classic2024]~^ ERROR: cannot move out of type
    #[cfg(any(stable2021, classic2021, structural2021))] let _: u32 = x;
    #[cfg(structural2024)] let _: &mut u32 = x;
}
