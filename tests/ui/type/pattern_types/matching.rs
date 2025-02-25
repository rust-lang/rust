#![feature(pattern_types, pattern_type_macro, structural_match)]

//@ check-pass

use std::marker::StructuralPartialEq;
use std::pat::pattern_type;

struct Thing(pattern_type!(u32 is 1..));

impl StructuralPartialEq for Thing {}
impl Eq for Thing {}
impl PartialEq for Thing {
    fn eq(&self, other: &Thing) -> bool {
        // Never called, pattern matching of struct-eq types
        // destructures the type and equates its fields.
        todo!()
    }
}

const TWO: Thing = Thing(unsafe { std::mem::transmute(2_u32) });

const _: () = match TWO {
    TWO => {}
    _ => unreachable!(),
};

fn main() {}
