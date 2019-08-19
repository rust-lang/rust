// build-pass (FIXME(62277): could be check-pass?)
pub trait Trait where Self::Out: std::fmt::Display {
    type Out;
}

fn main() {}
