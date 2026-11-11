pub trait X {
    const CONSTANT: u32;
    type Type;
    fn method(&self, s: String) -> Self::Type;
    fn method2(self: Box<Self>, s: String) -> Self::Type;
    fn method3(other: &Self, s: String) -> Self::Type;
    fn method4(&self, other: &Self) -> Self::Type;
    fn method5(self: &Box<Self>) -> Self::Type;
}
