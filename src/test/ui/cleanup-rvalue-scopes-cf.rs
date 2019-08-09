// Test that the borrow checker prevents pointers to temporaries
// with statement lifetimes from escaping.

use std::ops::Drop;

static mut FLAGS: u64 = 0;

struct StackBox<T> { f: T }
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
    let x1 = arg(&AddFlags(1)); //~ ERROR temporary value dropped while borrowed
    let x2 = AddFlags(1).get(); //~ ERROR temporary value dropped while borrowed
    let x3 = &*arg(&AddFlags(1)); //~ ERROR temporary value dropped while borrowed
    let ref x4 = *arg(&AddFlags(1)); //~ ERROR temporary value dropped while borrowed
    let &ref x5 = arg(&AddFlags(1)); //~ ERROR temporary value dropped while borrowed
    let x6 = AddFlags(1).get(); //~ ERROR temporary value dropped while borrowed
    let StackBox { f: x7 } = StackBox { f: AddFlags(1).get() };
    //~^ ERROR temporary value dropped while borrowed
    (x1, x2, x3, x4, x5, x6, x7);
}
