pub use internal::core::{Trait, Struct};

mod internal {
    pub mod core {
        pub struct Struct;
        impl Struct {
            pub fn init() -> Struct {
                Struct
            }
        }

        pub trait Trait {
            fn test(&self) {
                private();
            }
        }

        impl Trait for Struct {}

        fn private() { }
    }
}
