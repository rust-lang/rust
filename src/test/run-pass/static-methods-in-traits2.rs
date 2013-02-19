pub trait Number: NumConv {
    static pure fn from<T:Number>(n: T) -> Self;
}

pub impl Number for float {
    static pure fn from<T:Number>(n: T) -> float { n.to_float() }
}

pub trait NumConv {
    pure fn to_float(&self) -> float;
}

pub impl NumConv for float {
    pure fn to_float(&self) -> float { *self }
}

pub fn main() {
    let _: float = Number::from(0.0f);
}
