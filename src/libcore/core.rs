// Top-level, visible-everywhere definitions.

// Export various ubiquitous types, constructors, methods.

use option::{Some, None};
use Option = option::Option;
use result::{Result, Ok, Err};

use Path = path::Path;
use GenericPath = path::GenericPath;
use WindowsPath = path::WindowsPath;
use PosixPath = path::PosixPath;

use tuple::{TupleOps, ExtendedTupleOps};
use str::{StrSlice, UniqueStr};
use vec::{ConstVector, CopyableVector, ImmutableVector};
use vec::{ImmutableEqVector, ImmutableCopyableVector};
use iter::{BaseIter, ExtendedIter, EqIter, CopyableIter};
use iter::{CopyableOrderedIter, Times, TimesIx};
use num::Num;
use ptr::Ptr;
use to_str::ToStr;

export Path, WindowsPath, PosixPath, GenericPath;
export Option, Some, None;
export Result, Ok, Err;
export extensions;
// The following exports are the extension impls for numeric types
export Num, Times, TimesIx;
// The following exports are the common traits
export StrSlice, UniqueStr;
export ConstVector, CopyableVector, ImmutableVector;
export ImmutableEqVector, ImmutableCopyableVector, IterTraitExtensions;
export BaseIter, CopyableIter, CopyableOrderedIter, ExtendedIter, EqIter;
export TupleOps, ExtendedTupleOps;
export Ptr;
export ToStr;

// The following exports are the core operators and kinds
// The compiler has special knowlege of these so we must not duplicate them
// when compiling for testing
#[cfg(notest)]
use ops::{Const, Copy, Send, Owned};
#[cfg(notest)]
use ops::{Add, Sub, Mul, Div, Modulo, Neg, BitAnd, BitOr, BitXor};
#[cfg(notest)]
use ops::{Shl, Shr, Index};

#[cfg(notest)]
export Const, Copy, Send, Owned;
#[cfg(notest)]
export Add, Sub, Mul, Div, Modulo, Neg, BitAnd, BitOr, BitXor;
#[cfg(notest)]
export Shl, Shr, Index;

#[cfg(test)]
extern mod coreops(name = "core", vers = "0.4");

#[cfg(test)]
use coreops::ops::{Const, Copy, Send, Owned};
#[cfg(test)]
use coreops::ops::{Add, Sub, Mul, Div, Modulo, Neg, BitAnd, BitOr, BitXor};
#[cfg(test)]
use coreops::ops::{Shl, Shr, Index};


// Export the log levels as global constants. Higher levels mean
// more-verbosity. Error is the bottom level, default logging level is
// warn-and-below.

export error, warn, info, debug;

/// The error log level
const error : u32 = 0_u32;
/// The warning log level
const warn : u32 = 1_u32;
/// The info log level
const info : u32 = 2_u32;
/// The debug log level
const debug : u32 = 3_u32;

// A curious inner-module that's not exported that contains the binding
// 'core' so that macro-expanded references to core::error and such
// can be resolved within libcore.
#[doc(hidden)] // FIXME #3538
mod core {
    const error : u32 = 0_u32;
    const warn : u32 = 1_u32;
    const info : u32 = 2_u32;
    const debug : u32 = 3_u32;
}

// Similar to above. Some magic to make core testable.
#[cfg(test)]
mod std {
    extern mod std(vers = "0.4");
    use std::test;
}
