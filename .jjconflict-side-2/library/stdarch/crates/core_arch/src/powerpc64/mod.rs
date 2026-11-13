//! PowerPC 64
//!
//! The reference is the [64-Bit ELF V2 ABI Specification - Power
//! Architecture].
//!
//! [64-Bit ELF V2 ABI Specification - Power Architecture]: http://openpowerfoundation.org/wp-content/uploads/resources/leabi/leabi-20170510.pdf

mod vsx;

#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub use crate::core_arch::powerpc::*;

#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub use self::vsx::*;
