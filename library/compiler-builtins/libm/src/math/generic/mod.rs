//! Generic implementations that are shared by multiple types.
//!
//! Implementation and usage notes:
//!
//! * Generic functions are marked `#[inline]` because, even though generic functions are
//!   typically inlined, we seem to occasionally run into exceptions.
//! * Tests usually live wherever the functions are consumed (e.g. `src/ceil`) so they can be
//!   reused to test arch-specific implementations.

mod ceil;
mod copysign;
mod fabs;
mod fdim;
mod floor;
mod fma;
mod fma_wide;
mod fmax;
mod fmaximum;
mod fmaximum_num;
mod fmin;
mod fminimum;
mod fminimum_num;
mod fmod;
mod frexp;
mod ilogb;
mod rint;
mod round;
mod scalbn;
mod sqrt;
mod trunc;

pub use ceil::ceil_status;
pub use copysign::copysign;
pub use fabs::fabs;
pub use fdim::fdim;
pub use floor::floor_status;
pub use fma::fma_round;
pub use fma_wide::fma_wide_round;
pub use fmax::fmax;
pub use fmaximum::fmaximum;
pub use fmaximum_num::fmaximum_num;
pub use fmin::fmin;
pub use fminimum::fminimum;
pub use fminimum_num::fminimum_num;
pub use fmod::fmod;
pub use frexp::frexp;
pub use ilogb::ilogb;
pub use rint::rint_round;
pub use round::round;
pub use scalbn::scalbn;
#[cfg(test)]
pub use sqrt::SqrtHelper;
pub use sqrt::sqrt_round;
pub use trunc::trunc_status;
