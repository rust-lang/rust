pub trait Number: NumConv {
    fn from<T:Number>(n: T) -> Self;
}

impl Number for float {
    fn from<T:Number>(n: T) -> float { n.to_float() }
}

pub trait NumConv {
    fn to_float(&self) -> float;
}

impl NumConv for float {
    fn to_float(&self) -> float { *self }
}

pub fn main() {
    let _: float = Number::from(0.0f);
}
