//@ check-pass
#![feature(lazy_type_alias)]

use src::hidden_core;
mod src {
    mod aliases {
        use hidden_core::InternalStruct;
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
    }
}
fn main() {}
