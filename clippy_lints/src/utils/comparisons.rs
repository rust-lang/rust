// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utility functions about comparison operators.

#![deny(clippy::missing_docs_in_private_items)]

use crate::rustc::hir::{BinOpKind, Expr};

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
/// Represent a normalized comparison operator.
pub enum Rel {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `==`
    Eq,
    /// `!=`
    Ne,
}

/// Put the expression in the form  `lhs < rhs`, `lhs <= rhs`, `lhs == rhs` or
/// `lhs != rhs`.
pub fn normalize_comparison<'a>(op: BinOpKind, lhs: &'a Expr, rhs: &'a Expr) -> Option<(Rel, &'a Expr, &'a Expr)> {
    match op {
        BinOpKind::Lt => Some((Rel::Lt, lhs, rhs)),
        BinOpKind::Le => Some((Rel::Le, lhs, rhs)),
        BinOpKind::Gt => Some((Rel::Lt, rhs, lhs)),
        BinOpKind::Ge => Some((Rel::Le, rhs, lhs)),
        BinOpKind::Eq => Some((Rel::Eq, rhs, lhs)),
        BinOpKind::Ne => Some((Rel::Ne, rhs, lhs)),
        _ => None,
    }
}
