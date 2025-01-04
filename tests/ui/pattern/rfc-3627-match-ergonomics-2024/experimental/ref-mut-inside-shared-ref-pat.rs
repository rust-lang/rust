//@ edition: 2024
//@ run-rustfix
//@ revisions: classic structural
//! Tests for `&` patterns matched against `&mut` reference types where the inner pattern attempts
//! to bind by mutable reference.
#![allow(incomplete_features)]
#![cfg_attr(classic, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(&Some(ref mut x)) = &mut Some(Some(0)) {
        //~^ ERROR: cannot borrow as mutable inside an `&` pattern
        let _: &mut u8 = x;
    }

    if let &Some(Some(ref mut x)) = &mut Some(Some(0)) {
        //~^ ERROR: cannot borrow as mutable inside an `&` pattern
        let _: &mut u8 = x;
    }

    macro_rules! pat {
        ($var:ident) => { ref mut $var };
    }
    let &pat!(x) = &mut 0;
    //~^ ERROR: cannot borrow as mutable inside an `&` pattern
    let _: &mut u8 = x;

    let &(ref mut a, ref mut b) = &mut (true, false);
    //~^ ERROR: cannot borrow as mutable inside an `&` pattern
    //~| ERROR: cannot borrow as mutable inside an `&` pattern
    let _: &mut bool = a;
    let _: &mut bool = b;
}
