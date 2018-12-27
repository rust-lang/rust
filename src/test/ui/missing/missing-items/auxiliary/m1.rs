pub trait X {
    const CONSTANT: u32;
    type Type;
    fn method(&self, s: String) -> Self::Type;
}
