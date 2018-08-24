// ignore-compare-mode-nll

// Test that the borrow checker prevents pointers to temporaries
// with statement lifetimes from escaping.

use std::ops::Drop;

static mut FLAGS: u64 = 0;

struct Box<T> { f: T }
struct AddFlags { bits: u64 }

fn AddFlags(bits: u64) -> AddFlags {
    AddFlags { bits: bits }
}

fn arg(x: &AddFlags) -> &AddFlags {
    x
}

impl AddFlags {
    fn get(&self) -> &AddFlags {
        self
    }
}

pub fn main() {
    let _x = arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let _x = AddFlags(1).get(); //~ ERROR value does not live long enough
    let _x = &*arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let ref _x = *arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let &ref _x = arg(&AddFlags(1)); //~ ERROR value does not live long enough
    let _x = AddFlags(1).get(); //~ ERROR value does not live long enough
    let Box { f: _x } = Box { f: AddFlags(1).get() }; //~ ERROR value does not live long enough
}
