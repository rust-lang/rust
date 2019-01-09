#![crate_type="lib"]
#![deny(warnings)]

pub use src::aliases::B;
pub use src::hidden_core::make;

mod src {
    pub mod aliases {
        use super::hidden_core::A;
        pub type B = A;
    }

    pub mod hidden_core {
        use super::aliases::B;

        #[derive(Copy, Clone)]
        pub struct A;

        pub fn make() -> B { A }

        impl A {
            pub fn foo(&mut self) { println!("called foo"); }
        }
    }
}
