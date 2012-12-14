trait X {}

impl<A:Copy> A : X {}

struct S {
    x: int,
    drop {}
}

impl S : X {}

fn main(){}

