pub mod num {
    pub trait Num2 {
        pure fn from_int2(n: int) -> Self;
    }
}

pub mod float {
    impl ::num::Num2 for float {
        pure fn from_int2(n: int) -> float { return n as float;  }
    }
}
