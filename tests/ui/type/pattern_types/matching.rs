#![feature(pattern_types, pattern_type_macro, structural_match)]

//@ check-pass
//@ compile-flags: -Zvalidate-mir

use std::marker::StructuralPartialEq;
use std::pat::pattern_type;

struct Thing(pattern_type!(u32 is 1..));

impl StructuralPartialEq for Thing {}
impl PartialEq for Thing {
    fn eq(&self, other: &Thing) -> bool {
        unsafe { std::mem::transmute::<_, u32>(self.0) == std::mem::transmute::<_, u32>(other.0) }
    }
}

impl Eq for Thing {}

const TWO: Thing = Thing(2);

const _: () = match TWO {
    TWO => {}
    _ => unreachable!(),
};

fn main() {}
