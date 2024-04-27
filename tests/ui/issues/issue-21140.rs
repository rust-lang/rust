//@ check-pass
pub trait Trait where Self::Out: std::fmt::Display {
    type Out;
}

fn main() {}
