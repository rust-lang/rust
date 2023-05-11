#![crate_type="lib"]
#![deny(warnings)]
#![allow(dead_code)]

pub use src::aliases::B;
pub use src::hidden_core::make;

mod src {
    pub mod aliases {
        use super::hidden_core::A;
        pub type B = A<f32>;
    }

    pub mod hidden_core {
        use super::aliases::B;

        pub struct A<T> { t: T }

        pub fn make() -> B { A { t: 1.0 } }

        impl<T> A<T> {
            pub fn foo(&mut self) { println!("called foo"); }
        }
    }
}
