pub mod num {
    pub trait Num2 {
        fn from_int2(n: int) -> Self;
    }
}

pub mod f64 {
    impl ::num::Num2 for f64 {
        fn from_int2(n: int) -> f64 { return n as f64;  }
    }
}
