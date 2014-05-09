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

#![allow(missing_doc)]
#![doc(hidden)]

use option::Option;

pub enum Piece<'a> {
    String(&'a str),
    // FIXME(#8259): this shouldn't require the unit-value here
    CurrentArgument(()),
    Argument(Argument<'a>),
}

pub struct Argument<'a> {
    pub position: Position,
    pub format: FormatSpec,
    pub method: Option<&'a Method<'a>>
}

pub struct FormatSpec {
    pub fill: char,
    pub align: Alignment,
    pub flags: uint,
    pub precision: Count,
    pub width: Count,
}

#[deriving(Eq)]
pub enum Alignment {
    AlignLeft,
    AlignRight,
    AlignUnknown,
}

pub enum Count {
    CountIs(uint), CountIsParam(uint), CountIsNextParam, CountImplied,
}

pub enum Position {
    ArgumentNext, ArgumentIs(uint)
}

pub enum Flag {
    FlagSignPlus,
    FlagSignMinus,
    FlagAlternate,
    FlagSignAwareZeroPad,
}

pub enum Method<'a> {
    Plural(Option<uint>, &'a [PluralArm<'a>], &'a [Piece<'a>]),
    Select(&'a [SelectArm<'a>], &'a [Piece<'a>]),
}

pub enum PluralSelector {
    Keyword(PluralKeyword),
    Literal(uint),
}

pub enum PluralKeyword {
    Zero,
    One,
    Two,
    Few,
    Many,
}

pub struct PluralArm<'a> {
    pub selector: PluralSelector,
    pub result: &'a [Piece<'a>],
}

pub struct SelectArm<'a> {
    pub selector: &'a str,
    pub result: &'a [Piece<'a>],
}
