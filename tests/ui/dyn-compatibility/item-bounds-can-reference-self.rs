//@ check-pass

pub trait Foo {
    type X: PartialEq;
    type Y: PartialEq<Self::Y>;
    type Z: PartialEq<Self::Y>;
}

fn uwu(x: &dyn Foo<X = i32, Y = i32, Z = i32>) {}

fn main() {}
