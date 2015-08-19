#![feature(plugin)]
#![plugin(clippy)]

#![deny(unit_cmp)]

fn main() {
    // this is fine
    if true == false {
    }

    // this warns
    if { true; } == { false; } {  //~ERROR ==-comparison of unit values detected. This will always be true
    }

    if { true; } > { false; } {  //~ERROR >-comparison of unit values detected. This will always be false
    }
}
