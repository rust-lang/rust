#![crate_type = "rlib"]
//@ edition: 2024
//@ compile-flags: -Zunpretty=thir-flat
//@ check-pass

// Snapshot test capturing the original constant value recorded on array and
// slice pattern nodes expanded from constants by `const_to_pat`, which match
// lowering uses to compare the scrutinee against the constant as a whole.

const ARRAY: [u8; 4] = [1, 2, 3, 4];
const SLICE: &[u8] = b"ABCD";

pub fn array(x: [u8; 4]) {
    match x {
        ARRAY => {}
        _ => {}
    }
}

pub fn slice(x: &[u8]) {
    match x {
        SLICE => {}
        _ => {}
    }
}
