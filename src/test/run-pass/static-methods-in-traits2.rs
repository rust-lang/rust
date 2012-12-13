pub trait Number: NumConv {
    static pure fn from<T:Number>(n: T) -> self;
}

pub impl float: Number {
    static pure fn from<T:Number>(n: T) -> float { n.to_float() }
}

pub trait NumConv {
    pure fn to_float(&self) -> float;
}

pub impl float: NumConv {
    pure fn to_float(&self) -> float { *self }
}

fn main() {
    let _: float = Number::from(0.0f);
}