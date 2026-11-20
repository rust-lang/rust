pub mod m1 {
    pub trait Trait {
        fn method1(&self) {}
    }
    impl Trait for u8 {}
}
pub mod m2 {
    pub trait Trait {
        fn method2(&self) {}
    }
    impl Trait for u8 {}
}
pub mod m1_reexport {
    pub use crate::m1::Trait;
}
pub mod m2_reexport {
    pub use crate::m2::Trait;
}

pub mod ambig_reexport {
    pub use crate::m1::*;
    pub use crate::m2::*;
}
