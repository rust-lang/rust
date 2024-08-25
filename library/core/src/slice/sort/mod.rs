//! This module and the contained sub-modules contains the code for efficient and robust sort
//! implementations, as well as the domain adjacent implementation of `select_nth_unstable`.

pub mod stable;
pub mod unstable;

pub(crate) mod select;
#[cfg(not(feature = "optimize_for_size"))]
pub(crate) mod shared;
