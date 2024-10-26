#[macro_use]
pub mod macros;
mod float_traits;
mod int_traits;

pub use float_traits::Float;
pub use int_traits::{CastFrom, CastInto, DInt, HInt, Int, MinInt};
