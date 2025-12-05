//@ check-pass

#![allow(clippy::unit_arg)]

struct One {
    x: i32,
}
struct Two {
    x: i32,
}

struct Product {}

impl Product {
    pub fn a_method(self, _: ()) {}
}

fn from_array(_: [i32; 2]) -> Product {
    todo!()
}

pub fn main() {
    let one = One { x: 1 };
    let two = Two { x: 2 };

    let product = from_array([one.x, two.x]);
    product.a_method(<()>::default());
}
