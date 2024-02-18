//@ check-pass

pub fn main() {}

pub trait Iced {
    fn get(&self) -> &impl Sized;
}

/// Impl causes ICE
impl Iced for () {
    fn get(&self) -> &impl Sized {
        &()
    }
}
