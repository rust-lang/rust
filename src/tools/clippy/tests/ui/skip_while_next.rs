//@aux-build:option_helpers.rs

#![warn(clippy::skip_while_next)]
#![allow(clippy::disallowed_names, clippy::useless_vec)]

extern crate option_helpers;
use option_helpers::IteratorFalsePositives;

#[rustfmt::skip]
fn skip_while_next() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // Single-line case.
    let _ = v.iter().skip_while(|&x| *x < 0).next();
    //~^ skip_while_next

    // Multi-line case.
    let _ = v.iter().skip_while(|&x| {
    //~^ skip_while_next
                                *x < 0
                            }
                   ).next();

    // Check that hat we don't lint if the caller is not an `Iterator`.
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.skip_while().next();
}

fn main() {
    skip_while_next();
}
