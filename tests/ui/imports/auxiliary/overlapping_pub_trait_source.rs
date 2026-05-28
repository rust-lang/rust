/* This crate declares an item as both `prelude::*` and `m::Tr`.
 * The compiler should always suggest `m::Tr`. */

pub struct S;

pub mod prelude {
    pub use crate::m::Tr as _;
}

pub mod m {
    pub trait Tr { fn method(&self); }
    impl Tr for crate::S { fn method(&self) {} }
}
