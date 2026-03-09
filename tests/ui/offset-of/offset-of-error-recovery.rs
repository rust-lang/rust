use std::mem::offset_of;

struct S {
    x: (),
}

impl S {
    fn a() {
        offset_of!(Self, Self::x);
        //~^ ERROR offset_of expects dot-separated field and variant names
    }
}

fn main() {}
