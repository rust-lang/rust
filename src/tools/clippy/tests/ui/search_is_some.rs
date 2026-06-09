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
