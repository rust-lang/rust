// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Allows the buffering of lints for later.
//!
//! Since we cannot have a dependency on `librustc`, we implement some types here that are somewhat
//! redundant. Later, these types can be converted to types for use by the rest of the compiler.

use syntax::ast::NodeId;
use syntax_pos::MultiSpan;

/// Since we cannot import `LintId`s from `rustc::lint`, we define some Ids here which can later be
/// passed to `rustc::lint::Lint::from_parser_lint_id` to get a `rustc::lint::Lint`.
pub enum BufferedEarlyLintId {
    /// Usage of `?` as a macro separator is deprecated.
    QuestionMarkMacroSep,
}

/// Stores buffered lint info which can later be passed to `librustc`.
pub struct BufferedEarlyLint {
    /// The span of code that we are linting on.
   pub span: MultiSpan,

   /// The lint message.
   pub msg: String,

   /// The `NodeId` of the AST node that generated the lint.
   pub id: NodeId,

   /// A lint Id that can be passed to `rustc::lint::Lint::from_parser_lint_id`.
   pub lint_id: BufferedEarlyLintId,
}
