//! PowerPC 64
//!
//! The reference is the [64-Bit ELF V2 ABI Specification - Power Architecture].
//!
//! [64-Bit ELF V2 ABI Specification - Power Architecture]: http://openpowerfoundation.org/wp-content/uploads/resources/leabi/leabi-20170510.pdf

pub use coresimd::powerpc::*;

mod vsx;
pub use self::vsx::*;
