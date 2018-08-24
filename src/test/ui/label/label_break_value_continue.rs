#![feature(label_break_value)]
#![allow(unused_labels)]

// Simple continue pointing to an unlabeled break should yield in an error
fn continue_simple() {
    'b: {
        continue; //~ ERROR unlabeled `continue` inside of a labeled block
    }
}

// Labeled continue pointing to an unlabeled break should yield in an error
fn continue_labeled() {
    'b: {
        continue 'b; //~ ERROR `continue` pointing to a labeled block
    }
}

// Simple continue that would cross a labeled block should yield in an error
fn continue_crossing() {
    loop {
        'b: {
            continue; //~ ERROR unlabeled `continue` inside of a labeled block
        }
    }
}

pub fn main() {}
