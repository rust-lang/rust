#![feature(plugin)]
#![plugin(clippy)]

#[macro_use]
extern crate lazy_static;

use std::collections::HashMap;

#[deny(mut_mut)]
#[allow(unused_variables, unused_mut)]
fn main() {
    lazy_static! {
        static ref MUT_MAP : HashMap<usize, &'static str> = {
            let mut m = HashMap::new();
            let mut zero = &mut &mut "zero";
            m.insert(0, "zero");
            m
        };
        static ref MUT_COUNT : usize = MUT_MAP.len();
    }
    assert!(*MUT_COUNT == 1);
}
