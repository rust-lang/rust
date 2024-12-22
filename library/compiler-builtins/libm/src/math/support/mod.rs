#[macro_use]
pub mod macros;
mod float_traits;
mod hex_float;
mod int_traits;

#[allow(unused_imports)]
pub use float_traits::{Float, IntTy};
#[allow(unused_imports)]
pub use hex_float::{hf32, hf64};
pub use int_traits::{CastFrom, CastInto, DInt, HInt, Int, MinInt};
