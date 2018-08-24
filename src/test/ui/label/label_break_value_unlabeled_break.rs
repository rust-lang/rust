#![feature(label_break_value)]
#![allow(unused_labels)]

// Simple unlabeled break should yield in an error
fn unlabeled_break_simple() {
    'b: {
        break; //~ ERROR unlabeled `break` inside of a labeled block
    }
}

// Unlabeled break that would cross a labeled block should yield in an error
fn unlabeled_break_crossing() {
    loop {
        'b: {
            break; //~ ERROR unlabeled `break` inside of a labeled block
        }
    }
}

pub fn main() {}
