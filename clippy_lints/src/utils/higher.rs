//! This module contains functions for retrieve the original AST from lowered
//! `hir`.

#![deny(clippy::missing_docs_in_private_items)]

use crate::utils::{is_expn_of, match_def_path, match_qpath, paths, resolve_node};
use if_chain::if_chain;
use rustc::lint::LateContext;
use rustc::{hir, ty};
use syntax::ast;

/// Converts a hir binary operator to the corresponding `ast` type.
pub fn binop(op: hir::BinOpKind) -> ast::BinOpKind {
    match op {
        hir::BinOpKind::Eq => ast::BinOpKind::Eq,
        hir::BinOpKind::Ge => ast::BinOpKind::Ge,
        hir::BinOpKind::Gt => ast::BinOpKind::Gt,
        hir::BinOpKind::Le => ast::BinOpKind::Le,
        hir::BinOpKind::Lt => ast::BinOpKind::Lt,
        hir::BinOpKind::Ne => ast::BinOpKind::Ne,
        hir::BinOpKind::Or => ast::BinOpKind::Or,
        hir::BinOpKind::Add => ast::BinOpKind::Add,
        hir::BinOpKind::And => ast::BinOpKind::And,
        hir::BinOpKind::BitAnd => ast::BinOpKind::BitAnd,
        hir::BinOpKind::BitOr => ast::BinOpKind::BitOr,
        hir::BinOpKind::BitXor => ast::BinOpKind::BitXor,
        hir::BinOpKind::Div => ast::BinOpKind::Div,
        hir::BinOpKind::Mul => ast::BinOpKind::Mul,
        hir::BinOpKind::Rem => ast::BinOpKind::Rem,
        hir::BinOpKind::Shl => ast::BinOpKind::Shl,
        hir::BinOpKind::Shr => ast::BinOpKind::Shr,
        hir::BinOpKind::Sub => ast::BinOpKind::Sub,
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
pub fn range<'a, 'b, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'b hir::Expr) -> Option<Range<'b>> {
    /// Finds the field named `name` in the field. Always return `Some` for
    /// convenience.
    fn get_field<'c>(name: &str, fields: &'c [hir::Field]) -> Option<&'c hir::Expr> {
        let expr = &fields.iter().find(|field| field.ident.name.as_str() == name)?.expr;

        Some(expr)
    }

    let def_path = match cx.tables.expr_ty(expr).sty {
        ty::Adt(def, _) => cx.tcx.def_path(def.did),
        _ => return None,
    };

    // sanity checks for std::ops::RangeXXXX
    if def_path.data.len() != 3 {
        return None;
    }
    if def_path.data.get(0)?.data.as_interned_str().as_symbol() != sym!(ops) {
        return None;
    }
    if def_path.data.get(1)?.data.as_interned_str().as_symbol() != sym!(range) {
        return None;
    }
    let type_name = def_path.data.get(2)?.data.as_interned_str();
    let range_types = [
        "RangeFrom",
        "RangeFull",
        "RangeInclusive",
        "Range",
        "RangeTo",
        "RangeToInclusive",
    ];
    if !range_types.contains(&&*type_name.as_str()) {
        return None;
    }

    // The range syntax is expanded to literal paths starting with `core` or `std`
    // depending on
    // `#[no_std]`. Testing both instead of resolving the paths.

    match expr.node {
        hir::ExprKind::Path(ref path) => {
            if match_qpath(path, &paths::RANGE_FULL_STD) || match_qpath(path, &paths::RANGE_FULL) {
                Some(Range {
                    start: None,
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else {
                None
            }
        },
        hir::ExprKind::Call(ref path, ref args) => {
            if let hir::ExprKind::Path(ref path) = path.node {
                if match_qpath(path, &paths::RANGE_INCLUSIVE_STD_NEW) || match_qpath(path, &paths::RANGE_INCLUSIVE_NEW)
                {
                    Some(Range {
                        start: Some(&args[0]),
                        end: Some(&args[1]),
                        limits: ast::RangeLimits::Closed,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        },
        hir::ExprKind::Struct(ref path, ref fields, None) => {
            if match_qpath(path, &paths::RANGE_FROM_STD) || match_qpath(path, &paths::RANGE_FROM) {
                Some(Range {
                    start: Some(get_field("start", fields)?),
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else if match_qpath(path, &paths::RANGE_STD) || match_qpath(path, &paths::RANGE) {
                Some(Range {
                    start: Some(get_field("start", fields)?),
                    end: Some(get_field("end", fields)?),
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else if match_qpath(path, &paths::RANGE_TO_INCLUSIVE_STD) || match_qpath(path, &paths::RANGE_TO_INCLUSIVE)
            {
                Some(Range {
                    start: None,
                    end: Some(get_field("end", fields)?),
                    limits: ast::RangeLimits::Closed,
                })
            } else if match_qpath(path, &paths::RANGE_TO_STD) || match_qpath(path, &paths::RANGE_TO) {
                Some(Range {
                    start: None,
                    end: Some(get_field("end", fields)?),
                    limits: ast::RangeLimits::HalfOpen,
                })
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Checks if a `let` statement is from a `for` loop desugaring.
pub fn is_from_for_desugar(local: &hir::Local) -> bool {
    // This will detect plain for-loops without an actual variable binding:
    //
    // ```
    // for x in some_vec {
    //     // do stuff
    // }
    // ```
    if_chain! {
        if let Some(ref expr) = local.init;
        if let hir::ExprKind::Match(_, _, hir::MatchSource::ForLoopDesugar) = expr.node;
        then {
            return true;
        }
    }

    // This detects a variable binding in for loop to avoid `let_unit_value`
    // lint (see issue #1964).
    //
    // ```
    // for _ in vec![()] {
    //     // anything
    // }
    // ```
    if let hir::LocalSource::ForLoopDesugar = local.source {
        return true;
    }

    false
}

/// Recover the essential nodes of a desugared for loop:
/// `for pat in arg { body }` becomes `(pat, arg, body)`.
pub fn for_loop(expr: &hir::Expr) -> Option<(&hir::Pat, &hir::Expr, &hir::Expr)> {
    if_chain! {
        if let hir::ExprKind::Match(ref iterexpr, ref arms, hir::MatchSource::ForLoopDesugar) = expr.node;
        if let hir::ExprKind::Call(_, ref iterargs) = iterexpr.node;
        if iterargs.len() == 1 && arms.len() == 1 && arms[0].guard.is_none();
        if let hir::ExprKind::Loop(ref block, _, _) = arms[0].body.node;
        if block.expr.is_none();
        if let [ _, _, ref let_stmt, ref body ] = *block.stmts;
        if let hir::StmtKind::Local(ref local) = let_stmt.node;
        if let hir::StmtKind::Expr(ref expr) = body.node;
        then {
            return Some((&*local.pat, &iterargs[0], expr));
        }
    }
    None
}

/// Recover the essential nodes of a desugared while loop:
/// `while cond { body }` becomes `(cond, body)`.
pub fn while_loop(expr: &hir::Expr) -> Option<(&hir::Expr, &hir::Expr)> {
    if_chain! {
        if let hir::ExprKind::Loop(block, _, hir::LoopSource::While) = &expr.node;
        if let hir::Block { expr: Some(expr), .. } = &**block;
        if let hir::ExprKind::Match(cond, arms, hir::MatchSource::WhileDesugar) = &expr.node;
        if let hir::ExprKind::DropTemps(cond) = &cond.node;
        if let [arm, ..] = &arms[..];
        if let hir::Arm { body, .. } = arm;
        then {
            return Some((cond, body));
        }
    }
    None
}

/// Recover the essential nodes of a desugared if block
/// `if cond { then } else { els }` becomes `(cond, then, Some(els))`
pub fn if_block(expr: &hir::Expr) -> Option<(&hir::Expr, &hir::Expr, Option<&hir::Expr>)> {
    if let hir::ExprKind::Match(ref cond, ref arms, hir::MatchSource::IfDesugar { contains_else_clause }) = expr.node {
        let cond = if let hir::ExprKind::DropTemps(ref cond) = cond.node {
            cond
        } else {
            panic!("If block desugar must contain DropTemps");
        };
        let then = &arms[0].body;
        let els = if contains_else_clause {
            Some(&*arms[1].body)
        } else {
            None
        };
        Some((cond, then, els))
    } else {
        None
    }
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
pub fn vec_macro<'e>(cx: &LateContext<'_, '_>, expr: &'e hir::Expr) -> Option<VecArgs<'e>> {
    if_chain! {
        if let hir::ExprKind::Call(ref fun, ref args) = expr.node;
        if let hir::ExprKind::Path(ref path) = fun.node;
        if is_expn_of(fun.span, "vec").is_some();
        if let Some(fun_def_id) = resolve_node(cx, path, fun.hir_id).opt_def_id();
        then {
            return if match_def_path(cx, fun_def_id, &paths::VEC_FROM_ELEM) && args.len() == 2 {
                // `vec![elem; size]` case
                Some(VecArgs::Repeat(&args[0], &args[1]))
            }
            else if match_def_path(cx, fun_def_id, &paths::SLICE_INTO_VEC) && args.len() == 1 {
                // `vec![a, b, c]` case
                if_chain! {
                    if let hir::ExprKind::Box(ref boxed) = args[0].node;
                    if let hir::ExprKind::Array(ref args) = boxed.node;
                    then {
                        return Some(VecArgs::Vec(&*args));
                    }
                }

                None
            }
            else {
                None
            };
        }
    }

    None
}
