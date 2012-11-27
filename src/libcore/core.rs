// Top-level, visible-everywhere definitions.

// Export various ubiquitous types, constructors, methods.

pub use option::{Some, None};
pub use Option = option::Option;
pub use result::{Result, Ok, Err};

pub use Path = path::Path;
pub use GenericPath = path::GenericPath;
pub use WindowsPath = path::WindowsPath;
pub use PosixPath = path::PosixPath;

pub use tuple::{CopyableTuple, ImmutableTuple, ExtendedTupleOps};
pub use str::{StrSlice, Trimmable};
pub use vec::{ConstVector, CopyableVector, ImmutableVector};
pub use vec::{ImmutableEqVector, ImmutableCopyableVector};
pub use vec::{MutableVector, MutableCopyableVector};
pub use iter::{BaseIter, ExtendedIter, EqIter, CopyableIter};
pub use iter::{CopyableOrderedIter, CopyableNonstrictIter, Times};
pub use num::Num;
pub use ptr::Ptr;
pub use to_str::ToStr;

// The following exports are the core operators and kinds
// The compiler has special knowlege of these so we must not duplicate them
// when compiling for testing
#[cfg(notest)]
pub use ops::{Const, Copy, Send, Owned};
#[cfg(notest)]
pub use ops::{Drop};
#[cfg(notest)]
pub use ops::{Add, Sub, Mul, Div, Modulo, Neg, BitAnd, BitOr, BitXor};
#[cfg(notest)]
pub use ops::{Shl, Shr, Index};

#[cfg(test)]
extern mod coreops(name = "core", vers = "0.5");

#[cfg(test)]
pub use coreops::ops::{Const, Copy, Send, Owned};
#[cfg(test)]
pub use coreops::ops::{Drop};
#[cfg(test)]
pub use coreops::ops::{Add, Sub, Mul, Div, Modulo, Neg, BitAnd, BitOr};
#[cfg(test)]
pub use coreops::ops::{BitXor};
#[cfg(test)]
pub use coreops::ops::{Shl, Shr, Index};

#[cfg(notest)]
pub use clone::Clone;
#[cfg(test)]
pub use coreops::clone::Clone;

// Export the log levels as global constants. Higher levels mean
// more-verbosity. Error is the bottom level, default logging level is
// warn-and-below.

/// The error log level
pub const error : u32 = 1_u32;
/// The warning log level
pub const warn : u32 = 2_u32;
/// The info log level
pub const info : u32 = 3_u32;
/// The debug log level
pub const debug : u32 = 4_u32;

// A curious inner-module that's not exported that contains the binding
// 'core' so that macro-expanded references to core::error and such
// can be resolved within libcore.
#[doc(hidden)] // FIXME #3538
mod core {
    pub const error : u32 = 1_u32;
    pub const warn : u32 = 2_u32;
    pub const info : u32 = 3_u32;
    pub const debug : u32 = 4_u32;
}

// Similar to above. Some magic to make core testable.
#[cfg(test)]
mod std {
    extern mod std(vers = "0.5");
    pub use std::test;
}
