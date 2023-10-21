//! Traits for vectors with numeric elements.

mod float;
mod int;
mod uint;

mod sealed {
    pub trait Sealed {}
}

pub use float::*;
pub use int::*;
pub use uint::*;
