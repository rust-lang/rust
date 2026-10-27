//@error-in-other-file: aborted execution
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""

// Attempting to unwind out of `.use` is not UB,
// but is guaranteed to abort.

#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;

struct EvilClone;

impl Clone for EvilClone {
    fn clone(&self) -> Self {
        panic!();
    }
}

impl UseCloned for EvilClone {}

fn main() {
    let evil = EvilClone;
    let _copy = evil.use;
}
