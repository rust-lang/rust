//@ edition: 2024
//@ compile-flags: -Zunstable-options
#![allow(incomplete_features)]
#![feature(ref_pat_eat_one_layer_2024)]

pub fn main() {
    if let Some(&mut Some(&_)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&_)) = &Some(&mut Some(0)) {
        //~^ ERROR: mismatched types
    }
}
