//! This module contains functions for retrieve the original AST from lowered `hir`.

#![deny(missing_docs_in_private_items)]

use rustc::hir;
use rustc::lint::LateContext;
use syntax::ast;
use syntax::ptr::P;
use utils::{is_expn_of, match_path, paths};

/// Convert a hir binary operator to the corresponding `ast` type.
pub fn binop(op: hir::BinOp_) -> ast::BinOpKind {
    match op {
        hir::BiEq => ast::BinOpKind::Eq,
        hir::BiGe => ast::BinOpKind::Ge,
        hir::BiGt => ast::BinOpKind::Gt,
        hir::BiLe => ast::BinOpKind::Le,
        hir::BiLt => ast::BinOpKind::Lt,
        hir::BiNe => ast::BinOpKind::Ne,
        hir::BiOr => ast::BinOpKind::Or,
        hir::BiAdd => ast::BinOpKind::Add,
        hir::BiAnd => ast::BinOpKind::And,
        hir::BiBitAnd => ast::BinOpKind::BitAnd,
        hir::BiBitOr => ast::BinOpKind::BitOr,
        hir::BiBitXor => ast::BinOpKind::BitXor,
        hir::BiDiv => ast::BinOpKind::Div,
        hir::BiMul => ast::BinOpKind::Mul,
        hir::BiRem => ast::BinOpKind::Rem,
        hir::BiShl => ast::BinOpKind::Shl,
        hir::BiShr => ast::BinOpKind::Shr,
        hir::BiSub => ast::BinOpKind::Sub,
    }
}

/// Represent a range akin to `ast::ExprKind::Range`.
#[derive(Debug, Copy, Clone)]
pub struct Range<'a> {
    /// The lower bound of the range, or `None` for ranges such as `..X`.
    pub start: Option<&'a hir::Expr>,
    /// The upper bound of the range, or `None` for ranges such as `X..`.
    pub end: Option<&'a hir::Expr>,
    /// Whether the interval is open or closed.
    pub limits: ast::RangeLimits,
}

/// Higher a `hir` range to something similar to `ast::ExprKind::Range`.
pub fn range(expr: &hir::Expr) -> Option<Range> {
    /// Skip unstable blocks. To be removed when ranges get stable.
    fn unwrap_unstable(expr: &hir::Expr) -> &hir::Expr {
        if let hir::ExprBlock(ref block) = expr.node {
            if block.rules == hir::BlockCheckMode::PushUnstableBlock || block.rules == hir::BlockCheckMode::PopUnstableBlock {
                if let Some(ref expr) = block.expr {
                    return expr;
                }
            }
        }

        expr
    }

    /// Find the field named `name` in the field. Always return `Some` for convenience.
    fn get_field<'a>(name: &str, fields: &'a [hir::Field]) -> Option<&'a hir::Expr> {
        let expr = &fields.iter()
                          .find(|field| field.name.node.as_str() == name)
                          .unwrap_or_else(|| panic!("missing {} field for range", name))
                          .expr;

        Some(unwrap_unstable(expr))
    }

    // The range syntax is expanded to literal paths starting with `core` or `std` depending on
    // `#[no_std]`. Testing both instead of resolving the paths.

    match unwrap_unstable(expr).node {
        hir::ExprPath(None, ref path) => {
            if match_path(path, &paths::RANGE_FULL_STD) || match_path(path, &paths::RANGE_FULL) {
                Some(Range {
                    start: None,
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else {
                None
            }
        }
        hir::ExprStruct(ref path, ref fields, None) => {
            if match_path(path, &paths::RANGE_FROM_STD) || match_path(path, &paths::RANGE_FROM) {
                Some(Range {
                    start: get_field("start", fields),
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else if match_path(path, &paths::RANGE_INCLUSIVE_NON_EMPTY_STD) ||
               match_path(path, &paths::RANGE_INCLUSIVE_NON_EMPTY) {
                Some(Range {
                    start: get_field("start", fields),
                    end: get_field("end", fields),
                    limits: ast::RangeLimits::Closed,
                })
            } else if match_path(path, &paths::RANGE_STD) || match_path(path, &paths::RANGE) {
                Some(Range {
                    start: get_field("start", fields),
                    end: get_field("end", fields),
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else if match_path(path, &paths::RANGE_TO_INCLUSIVE_STD) || match_path(path, &paths::RANGE_TO_INCLUSIVE) {
                Some(Range {
                    start: None,
                    end: get_field("end", fields),
                    limits: ast::RangeLimits::Closed,
                })
            } else if match_path(path, &paths::RANGE_TO_STD) || match_path(path, &paths::RANGE_TO) {
                Some(Range {
                    start: None,
                    end: get_field("end", fields),
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Checks if a `let` decl is from a `for` loop desugaring.
pub fn is_from_for_desugar(decl: &hir::Decl) -> bool {
    if_let_chain! {[
        let hir::DeclLocal(ref loc) = decl.node,
        let Some(ref expr) = loc.init,
        let hir::ExprMatch(_, _, hir::MatchSource::ForLoopDesugar) = expr.node,
    ], {
        return true;
    }}
    false
}

/// Recover the essential nodes of a desugared for loop:
/// `for pat in arg { body }` becomes `(pat, arg, body)`.
pub fn for_loop(expr: &hir::Expr) -> Option<(&hir::Pat, &hir::Expr, &hir::Expr)> {
    if_let_chain! {[
        let hir::ExprMatch(ref iterexpr, ref arms, _) = expr.node,
        let hir::ExprCall(_, ref iterargs) = iterexpr.node,
        iterargs.len() == 1 && arms.len() == 1 && arms[0].guard.is_none(),
        let hir::ExprLoop(ref block, _) = arms[0].body.node,
        block.stmts.is_empty(),
        let Some(ref loopexpr) = block.expr,
        let hir::ExprMatch(_, ref innerarms, hir::MatchSource::ForLoopDesugar) = loopexpr.node,
        innerarms.len() == 2 && innerarms[0].pats.len() == 1,
        let hir::PatKind::TupleStruct(_, ref somepats, _) = innerarms[0].pats[0].node,
        somepats.len() == 1
    ], {
        return Some((&somepats[0],
                     &iterargs[0],
                     &innerarms[0].body));
    }}
    None
}

/// Represent the pre-expansion arguments of a `vec!` invocation.
pub enum VecArgs<'a> {
    /// `vec![elem; len]`
    Repeat(&'a P<hir::Expr>, &'a P<hir::Expr>),
    /// `vec![a, b, c]`
    Vec(&'a [P<hir::Expr>]),
}

/// Returns the arguments of the `vec!` macro if this expression was expanded from `vec!`.
pub fn vec_macro<'e>(cx: &LateContext, expr: &'e hir::Expr) -> Option<VecArgs<'e>> {
    if_let_chain!{[
        let hir::ExprCall(ref fun, ref args) = expr.node,
        let hir::ExprPath(_, ref path) = fun.node,
        is_expn_of(cx, fun.span, "vec").is_some()
    ], {
        return if match_path(path, &paths::VEC_FROM_ELEM) && args.len() == 2 {
            // `vec![elem; size]` case
            Some(VecArgs::Repeat(&args[0], &args[1]))
        }
        else if match_path(path, &["into_vec"]) && args.len() == 1 {
            // `vec![a, b, c]` case
            if_let_chain!{[
                let hir::ExprBox(ref boxed) = args[0].node,
                let hir::ExprVec(ref args) = boxed.node
            ], {
                return Some(VecArgs::Vec(&*args));
            }}

            None
        }
        else {
            None
        };
    }}

    None
}
