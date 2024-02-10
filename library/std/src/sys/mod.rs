/// The PAL (platform abstraction layer) contains platform-specific abstractions
/// for implementing the features in the other submodules, e.g. UNIX file
/// descriptors.
mod pal;

mod personality;

pub mod cmath;
pub mod os_str;
pub mod path;

// FIXME(117276): remove this, move feature implementations into individual
//                submodules.
pub use pal::*;
