#![feature(plugin)]
#![plugin(clippy)]

#![warn(unit_cmp)]
#![allow(no_effect, unnecessary_operation)]

#[derive(PartialEq)]
pub struct ContainsUnit(()); // should be fine

fn main() {
    // this is fine
    if true == false {
    }

    // this warns
    if { true; } == { false; } {
    }

    if { true; } > { false; } {
    }
}
