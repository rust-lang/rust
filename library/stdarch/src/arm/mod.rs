//! ARM intrinsics.
pub use self::v6::*;
pub use self::v7::*;
#[cfg(target_arch = "aarch64")]
pub use self::v8::*;

mod v6;
mod v7;
#[cfg(target_arch = "aarch64")]
mod v8;
