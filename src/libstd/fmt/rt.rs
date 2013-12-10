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

#[allow(missing_doc)];
#[doc(hidden)];

use either::Either;
use fmt::parse;
use option::Option;

pub enum Piece<'a> {
    String(&'a str),
    // FIXME(#8259): this shouldn't require the unit-value here
    CurrentArgument(()),
    Argument(Argument<'a>),
}

pub struct Argument<'a> {
    position: Position,
    format: FormatSpec,
    method: Option<&'a Method<'a>>
}

pub struct FormatSpec {
    fill: char,
    align: parse::Alignment,
    flags: uint,
    precision: Count,
    width: Count,
}

pub enum Count {
    CountIs(uint), CountIsParam(uint), CountIsNextParam, CountImplied,
}

pub enum Position {
    ArgumentNext, ArgumentIs(uint)
}

pub enum Method<'a> {
    Plural(Option<uint>, &'a [PluralArm<'a>], &'a [Piece<'a>]),
    Select(&'a [SelectArm<'a>], &'a [Piece<'a>]),
}

pub struct PluralArm<'a> {
    selector: Either<parse::PluralKeyword, uint>,
    result: &'a [Piece<'a>],
}

pub struct SelectArm<'a> {
    selector: &'a str,
    result: &'a [Piece<'a>],
}
