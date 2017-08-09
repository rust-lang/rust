//! This module contains functions for retrieve the original AST from lowered
//! `hir`.

#![deny(missing_docs_in_private_items)]

use rustc::hir;
use rustc::lint::LateContext;
use syntax::ast;
use utils::{is_expn_of, match_path, match_def_path, resolve_node, paths};

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
    /// Find the field named `name` in the field. Always return `Some` for
    /// convenience.
    fn get_field<'a>(name: &str, fields: &'a [hir::Field]) -> Option<&'a hir::Expr> {
        let expr = &fields
            .iter()
            .find(|field| field.name.node == name)
            .unwrap_or_else(|| panic!("missing {} field for range", name))
            .expr;

        Some(expr)
    }

    // The range syntax is expanded to literal paths starting with `core` or `std`
    // depending on
    // `#[no_std]`. Testing both instead of resolving the paths.

    match expr.node {
        hir::ExprPath(ref path) => {
            if match_path(path, &paths::RANGE_FULL_STD) || match_path(path, &paths::RANGE_FULL) {
                Some(Range {
                    start: None,
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else {
                None
            }
        },
        hir::ExprStruct(ref path, ref fields, None) => {
            if match_path(path, &paths::RANGE_FROM_STD) || match_path(path, &paths::RANGE_FROM) {
                Some(Range {
                    start: get_field("start", fields),
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else if match_path(path, &paths::RANGE_INCLUSIVE_STD) || match_path(path, &paths::RANGE_INCLUSIVE) {
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
        },
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
        let hir::ExprMatch(ref iterexpr, ref arms, hir::MatchSource::ForLoopDesugar) = expr.node,
        let hir::ExprCall(_, ref iterargs) = iterexpr.node,
        iterargs.len() == 1 && arms.len() == 1 && arms[0].guard.is_none(),
        let hir::ExprLoop(ref block, _, _) = arms[0].body.node,
        block.expr.is_none(),
        let [ _, _, ref let_stmt, ref body ] = *block.stmts,
        let hir::StmtDecl(ref decl, _) = let_stmt.node,
        let hir::DeclLocal(ref decl) = decl.node,
        let hir::StmtExpr(ref expr, _) = body.node,
    ], {
        return Some((&*decl.pat, &iterargs[0], expr));
    }}
    None
}

/// Represent the pre-expansion arguments of a `vec!` invocation.
pub enum VecArgs<'a> {
    /// `vec![elem; len]`
    Repeat(&'a hir::Expr, &'a hir::Expr),
    /// `vec![a, b, c]`
    Vec(&'a [hir::Expr]),
}

/// Returns the arguments of the `vec!` macro if this expression was expanded
/// from `vec!`.
pub fn vec_macro<'e>(cx: &LateContext, expr: &'e hir::Expr) -> Option<VecArgs<'e>> {
    if_let_chain!{[
        let hir::ExprCall(ref fun, ref args) = expr.node,
        let hir::ExprPath(ref path) = fun.node,
        is_expn_of(fun.span, "vec").is_some(),
    ], {
        let fun_def = resolve_node(cx, path, fun.id);
        return if match_def_path(cx.tcx, fun_def.def_id(), &paths::VEC_FROM_ELEM) && args.len() == 2 {
            // `vec![elem; size]` case
            Some(VecArgs::Repeat(&args[0], &args[1]))
        }
        else if match_def_path(cx.tcx, fun_def.def_id(), &paths::SLICE_INTO_VEC) && args.len() == 1 {
            // `vec![a, b, c]` case
            if_let_chain!{[
                let hir::ExprBox(ref boxed) = args[0].node,
                let hir::ExprArray(ref args) = boxed.node
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
