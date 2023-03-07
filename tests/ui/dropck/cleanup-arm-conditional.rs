// run-pass

#![allow(stable_features)]
#![allow(unused_imports)]
// Test that cleanup scope for temporaries created in a match
// arm is confined to the match arm itself.

// pretty-expanded FIXME #23616

#![feature(os)]

use std::os;

struct Test { x: isize }

impl Test {
    fn get_x(&self) -> Option<Box<isize>> {
        Some(Box::new(self.x))
    }
}

fn do_something(t: &Test) -> isize {

    // The cleanup scope for the result of `t.get_x()` should be the
    // arm itself and not the match, otherwise we'll (potentially) get
    // a crash trying to free an uninitialized stack slot.

    match t {
        &Test { x: 2 } if t.get_x().is_some() => {
            t.x * 2
        }
        _ => { 22 }
    }
}

pub fn main() {
    let t = Test { x: 1 };
    do_something(&t);
}
