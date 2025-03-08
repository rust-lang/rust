#![feature(pattern_types, pattern_type_macro, structural_match)]

//@ check-pass

use std::marker::StructuralPartialEq;
use std::pat::pattern_type;

#[derive(PartialEq)]
struct Thing(pattern_type!(u32 is 1..));

impl Eq for Thing {}

const TWO: Thing = Thing(2);

const _: () = match TWO {
    TWO => {}
    _ => unreachable!(),
};

fn main() {}
