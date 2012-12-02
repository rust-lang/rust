
pub mod num {
    pub trait Num2 {
        static pure fn from_int2(n: int) -> self;
    }
}

pub mod float {
    impl float: num::Num2 {
        #[inline]
        static pure fn from_int2(n: int) -> float { return n as float;  }
    }
}

