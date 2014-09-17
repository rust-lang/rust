// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is an internal module used by the ifmt! runtime. These structures are
//! emitted to static arrays to precompile format strings ahead of time.
//!
//! These definitions are similar to their `ct` equivalents, but differ in that
//! these can be statically allocated and are slightly optimized for the runtime

#[doc(hidden)]
pub struct Argument<'a> {
    pub position: Position,
    pub format: FormatSpec,
}

#[doc(hidden)]
pub struct FormatSpec {
    pub fill: char,
    pub align: Alignment,
    pub flags: uint,
    pub precision: Count,
    pub width: Count,
}

/// Possible alignments that can be requested as part of a formatting directive.
#[deriving(PartialEq)]
pub enum Alignment {
    /// Indication that contents should be left-aligned.
    AlignLeft,
    /// Indication that contents should be right-aligned.
    AlignRight,
    /// Indication that contents should be center-aligned.
    AlignCenter,
    /// No alignment was requested.
    AlignUnknown,
}

#[doc(hidden)]
pub enum Count {
    CountIs(uint), CountIsParam(uint), CountIsNextParam, CountImplied,
}

#[doc(hidden)]
pub enum Position {
    ArgumentNext, ArgumentIs(uint)
}

/// Flags which can be passed to formatting via a directive.
///
/// These flags are discovered through the `flags` field of the `Formatter`
/// structure. The flag in that structure is a union of these flags into a
/// `uint` where each flag's discriminant is the corresponding bit.
pub enum Flag {
    /// A flag which enables number formatting to always print the sign of a
    /// number.
    FlagSignPlus,
    /// Currently not a used flag
    FlagSignMinus,
    /// Indicates that the "alternate formatting" for a type should be used.
    ///
    /// The meaning of this flag is type-specific.
    FlagAlternate,
    /// Indicates that padding should be done with a `0` character as well as
    /// being aware of the sign to be printed.
    FlagSignAwareZeroPad,
}
