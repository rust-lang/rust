#![no_std]
#![warn(clippy::manual_take)]

pub fn manual_mem_take_should_reference_core() {
    let mut x = true;

    let _lint_negated = if x {
        //~^ manual_take
        x = false;
        false
    } else {
        true
    };

    let _lint = if x {
        //~^ manual_take
        x = false;
        true
    } else {
        false
    };
}
