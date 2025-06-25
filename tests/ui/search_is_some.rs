//@aux-build:option_helpers.rs
#![warn(clippy::search_is_some)]
#![allow(clippy::manual_pattern_char_comparison)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
extern crate option_helpers;
use option_helpers::IteratorFalsePositives;
//@no-rustfix
#[rustfmt::skip]
fn main() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];
    let y = &&42;


    // Check `find().is_some()`, multi-line case.
    let _ = v.iter().find(|&x| {
    //~^ search_is_some
                              *x < 0
                          }
                   ).is_some();

    // Check `position().is_some()`, multi-line case.
    let _ = v.iter().position(|&x| {
    //~^ search_is_some
                                  x < 0
                              }
                   ).is_some();

    // Check `rposition().is_some()`, multi-line case.
    let _ = v.iter().rposition(|&x| {
    //~^ search_is_some
                                   x < 0
                               }
                   ).is_some();

    // Check that we don't lint if the caller is not an `Iterator` or string
    let falsepos = IteratorFalsePositives { foo: 0 };
    let _ = falsepos.find().is_some();
    let _ = falsepos.position().is_some();
    let _ = falsepos.rposition().is_some();
    // check that we don't lint if `find()` is called with
    // `Pattern` that is not a string
    let _ = "hello world".find(|c: char| c == 'o' || c == 'l').is_some();

    let some_closure = |x: &u32| *x == 0;
    let _ = (0..1).find(some_closure).is_some();
    //~^ search_is_some
}

#[rustfmt::skip]
fn is_none() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];
    let y = &&42;


    // Check `find().is_none()`, multi-line case.
    let _ = v.iter().find(|&x| {
    //~^ search_is_some
                              *x < 0
                          }
                   ).is_none();

    // Check `position().is_none()`, multi-line case.
    let _ = v.iter().position(|&x| {
    //~^ search_is_some
                                  x < 0
                              }
                   ).is_none();

    // Check `rposition().is_none()`, multi-line case.
    let _ = v.iter().rposition(|&x| {
    //~^ search_is_some
                                   x < 0
                               }
                   ).is_none();

    // Check that we don't lint if the caller is not an `Iterator` or string
    let falsepos = IteratorFalsePositives { foo: 0 };
    let _ = falsepos.find().is_none();
    let _ = falsepos.position().is_none();
    let _ = falsepos.rposition().is_none();
    // check that we don't lint if `find()` is called with
    // `Pattern` that is not a string
    let _ = "hello world".find(|c: char| c == 'o' || c == 'l').is_none();

    let some_closure = |x: &u32| *x == 0;
    let _ = (0..1).find(some_closure).is_none();
    //~^ search_is_some
}

#[allow(clippy::match_like_matches_macro)]
fn issue15102() {
    let values = [None, Some(3)];
    let has_even = values
        //~^ search_is_some
        .iter()
        .find(|v| match v {
            Some(x) if x % 2 == 0 => true,
            _ => false,
        })
        .is_some();

    println!("{has_even}");
}
