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

#![stable]

#[cfg(stage0)] pub use self::Position::*;

#[cfg(stage0)] pub use self::Alignment::Left as AlignLeft;
#[cfg(stage0)] pub use self::Alignment::Right as AlignRight;
#[cfg(stage0)] pub use self::Alignment::Center as AlignCenter;
#[cfg(stage0)] pub use self::Alignment::Unknown as AlignUnknown;
#[cfg(stage0)] pub use self::Count::Is as CountIs;
#[cfg(stage0)] pub use self::Count::Implied as CountImplied;
#[cfg(stage0)] pub use self::Count::Param as CountIsParam;
#[cfg(stage0)] pub use self::Count::NextParam as CountIsNextParam;
#[cfg(stage0)] pub use self::Position::Next as ArgumentNext;
#[cfg(stage0)] pub use self::Position::At as ArgumentIs;

// SNAP 9e4e524
#[derive(Copy)]
#[cfg(not(stage0))]
#[stable]
pub struct Argument {
    #[stable]
    pub position: Position,
    #[stable]
    pub format: FormatSpec,
}
#[derive(Copy)]
#[cfg(stage0)]
pub struct Argument<'a> {
    pub position: Position,
    pub format: FormatSpec,
}

#[derive(Copy)]
#[stable]
pub struct FormatSpec {
    #[stable]
    pub fill: char,
    #[stable]
    pub align: Alignment,
    #[stable]
    pub flags: uint,
    #[stable]
    pub precision: Count,
    #[stable]
    pub width: Count,
}

/// Possible alignments that can be requested as part of a formatting directive.
#[derive(Copy, PartialEq)]
#[stable]
pub enum Alignment {
    /// Indication that contents should be left-aligned.
    #[stable]
    Left,
    /// Indication that contents should be right-aligned.
    #[stable]
    Right,
    /// Indication that contents should be center-aligned.
    #[stable]
    Center,
    /// No alignment was requested.
    #[stable]
    Unknown,
}

#[derive(Copy)]
#[stable]
pub enum Count {
    #[stable]
    Is(usize),
    #[stable]
    Param(usize),
    #[stable]
    NextParam,
    #[stable]
    Implied,
}

#[derive(Copy)]
#[stable]
pub enum Position {
    #[stable]
    Next,
    #[stable]
    At(usize)
}
