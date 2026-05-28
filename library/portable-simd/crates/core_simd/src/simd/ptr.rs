//! Traits for vectors of pointers.

mod const_ptr;
mod mut_ptr;

mod sealed {
    pub trait Sealed {}
}

pub use const_ptr::*;
pub use mut_ptr::*;
