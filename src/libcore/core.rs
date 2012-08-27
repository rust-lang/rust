// Top-level, visible-everywhere definitions.

// Export various ubiquitous types, constructors, methods.

import option::{Some, None};
import Option = option::Option;
// XXX: snapshot rustc is generating code that wants lower-case option
#[cfg(stage0)]
import option = option::Option;

import result::{Result, Ok, Err};

import Path = path::Path;
import GenericPath = path::GenericPath;
import WindowsPath = path::WindowsPath;
import PosixPath = path::PosixPath;

import tuple::{TupleOps, ExtendedTupleOps};
import str::{StrSlice, UniqueStr};
import vec::{ConstVector, CopyableVector, ImmutableVector};
import vec::{ImmutableEqVector, ImmutableCopyableVector};
import iter::{BaseIter, ExtendedIter, EqIter, CopyableIter};
import iter::{CopyableOrderedIter, Times, TimesIx};
import num::Num;
import ptr::Ptr;
import to_str::ToStr;

export Path, WindowsPath, PosixPath, GenericPath;
export Option, Some, None, unreachable;
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
import ops::{const, copy, send, owned};
#[cfg(notest)]
import ops::{add, sub, mul, div, modulo, neg, bitand, bitor, bitxor};
#[cfg(notest)]
import ops::{shl, shr, index};

#[cfg(notest)]
export const, copy, send, owned;
#[cfg(notest)]
export add, sub, mul, div, modulo, neg, bitand, bitor, bitxor;
#[cfg(notest)]
export shl, shr, index;

#[cfg(test)]
use coreops(name = "core", vers = "0.3");

#[cfg(test)]
import coreops::ops::{const, copy, send, owned};
#[cfg(test)]
import coreops::ops::{add, sub, mul, div, modulo, neg, bitand, bitor, bitxor};
#[cfg(test)]
import coreops::ops::{shl, shr, index};


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
mod core {
    const error : u32 = 0_u32;
    const warn : u32 = 1_u32;
    const info : u32 = 2_u32;
    const debug : u32 = 3_u32;
}

// Similar to above. Some magic to make core testable.
#[cfg(test)]
mod std {
    use std(vers = "0.3");
    import std::test;
}

/**
 * A standard function to use to indicate unreachable code. Because the
 * function is guaranteed to fail typestate will correctly identify
 * any code paths following the appearance of this function as unreachable.
 */
fn unreachable() -> ! {
    fail ~"Internal error: entered unreachable code";
}

