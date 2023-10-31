#![deny(clippy::mut_mut, clippy::zero_ptr)]
#![allow(dead_code)]

// FIXME: compiletest + extern crates doesn't work together. To make this test work, it would need
// the following three lines and the lazy_static crate.
//
//     #[macro_use]
//     extern crate lazy_static;
//     use std::collections::HashMap;

/// ensure that we don't suggest `is_null` inside constants
/// FIXME: once const fn is stable, suggest these functions again in constants

const BAA: *const i32 = 0 as *const i32;
static mut BAR: *const i32 = BAA;
static mut FOO: *const i32 = 0 as *const i32;

#[allow(unused_variables, unused_mut)]
fn main() {
    /*
    lazy_static! {
        static ref MUT_MAP : HashMap<usize, &'static str> = {
            let mut m = HashMap::new();
            m.insert(0, "zero");
            m
        };
        static ref MUT_COUNT : usize = MUT_MAP.len();
    }
    assert_eq!(*MUT_COUNT, 1);
    */
    // FIXME: don't lint in array length, requires `check_body`
    //let _ = [""; (42.0 < f32::NAN) as usize];
}
