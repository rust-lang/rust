trait X {}

impl<A:Copy> X for A {}

struct S {
    x: int,
    drop {}
}

impl X for S {}

pub fn main(){}

