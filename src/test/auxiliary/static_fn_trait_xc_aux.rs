pub mod num {
    pub trait Num2 {
        fn from_int2(n: int) -> Self;
    }
}

pub mod float {
    impl ::num::Num2 for float {
        fn from_int2(n: int) -> float { return n as float;  }
    }
}
