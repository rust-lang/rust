pub mod num {
    pub trait Num2 {
        fn from_int2(n: isize) -> Self;
    }
}

pub mod f64 {
    impl crate::num::Num2 for f64 {
        fn from_int2(n: isize) -> f64 { return n as f64;  }
    }
}
