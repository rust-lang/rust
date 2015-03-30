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

#![stable(feature = "rust1", since = "1.0.0")]

#[derive(Copy, Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Argument {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub position: Position,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub format: FormatSpec,
}

#[derive(Copy, Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct FormatSpec {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fill: char,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub align: Alignment,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub flags: u32,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub precision: Count,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub width: Count,
}

/// Possible alignments that can be requested as part of a formatting directive.
#[derive(Copy, Clone, PartialEq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Alignment {
    /// Indication that contents should be left-aligned.
    #[stable(feature = "rust1", since = "1.0.0")]
    Left,
    /// Indication that contents should be right-aligned.
    #[stable(feature = "rust1", since = "1.0.0")]
    Right,
    /// Indication that contents should be center-aligned.
    #[stable(feature = "rust1", since = "1.0.0")]
    Center,
    /// No alignment was requested.
    #[stable(feature = "rust1", since = "1.0.0")]
    Unknown,
}

#[derive(Copy, Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Count {
    #[stable(feature = "rust1", since = "1.0.0")]
    Is(usize),
    #[stable(feature = "rust1", since = "1.0.0")]
    Param(usize),
    #[stable(feature = "rust1", since = "1.0.0")]
    NextParam,
    #[stable(feature = "rust1", since = "1.0.0")]
    Implied,
}

#[derive(Copy, Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Position {
    #[stable(feature = "rust1", since = "1.0.0")]
    Next,
    #[stable(feature = "rust1", since = "1.0.0")]
    At(usize)
}
