//! Auxilary file for testing `dead_code` lint. This crate is compiled as a library and exposes
//! aliased types. When used externally, there should not be warnings of `dead_code`
//!
//! Issue: <https://github.com/rust-lang/rust/issues/14421>

// Expose internal types to be used in external test
pub use src::aliases::ExposedType;
pub use src::hidden_core::new;

mod src {
    pub mod aliases {
        use super::hidden_core::InternalStruct;
        pub type ExposedType = InternalStruct<f32>;
    }

    pub mod hidden_core {
        use super::aliases::ExposedType;

        pub struct InternalStruct<T> {
            _x: T,
        }

        pub fn new() -> ExposedType {
            InternalStruct { _x: 1.0 }
        }

        impl<T> InternalStruct<T> {
            pub fn foo(&mut self) {
                println!("called foo");
            }
        }
    }
}
