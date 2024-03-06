//@ build-pass
#![deny(unused)]

pub struct F;
pub struct B;

pub enum E {
    Foo(F),
    Bar(B),
}

fn main() {
    let _ = E::Foo(F);
}
