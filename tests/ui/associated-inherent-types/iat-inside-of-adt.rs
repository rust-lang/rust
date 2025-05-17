//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo {
    bar: Self::Bar,
}
impl Foo {
    pub type Bar = usize;
}

fn main() {
    Foo {
        bar: 10_usize,
    };
}
