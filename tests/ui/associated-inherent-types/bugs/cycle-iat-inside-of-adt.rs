// known-bug: #108491

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
// FIXME(inherent_associated_types): This should pass.

struct Foo {
    bar: Self::Bar,
}
impl Foo {
    pub type Bar = usize;
}

fn main() {}
