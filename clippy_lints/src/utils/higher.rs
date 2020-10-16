//! This module contains functions for retrieve the original AST from lowered
//! `hir`.

#![deny(clippy::missing_docs_in_private_items)]

use crate::utils::{is_expn_of, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_hir as hir;
use rustc_hir::{BorrowKind, Expr, ExprKind, StmtKind, UnOp};
use rustc_lint::LateContext;

/// Converts a hir binary operator to the corresponding `ast` type.
#[must_use]
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
    pub start: Option<&'a hir::Expr<'a>>,
    /// The upper bound of the range, or `None` for ranges such as `X..`.
    pub end: Option<&'a hir::Expr<'a>>,
    /// Whether the interval is open or closed.
    pub limits: ast::RangeLimits,
}

/// Higher a `hir` range to something similar to `ast::ExprKind::Range`.
pub fn range<'a>(expr: &'a hir::Expr<'_>) -> Option<Range<'a>> {
    /// Finds the field named `name` in the field. Always return `Some` for
    /// convenience.
    fn get_field<'c>(name: &str, fields: &'c [hir::Field<'_>]) -> Option<&'c hir::Expr<'c>> {
        let expr = &fields.iter().find(|field| field.ident.name.as_str() == name)?.expr;

        Some(expr)
    }

    match expr.kind {
        hir::ExprKind::Call(ref path, ref args)
            if matches!(
                path.kind,
                hir::ExprKind::Path(hir::QPath::LangItem(hir::LangItem::RangeInclusiveNew, _))
            ) =>
        {
            Some(Range {
                start: Some(&args[0]),
                end: Some(&args[1]),
                limits: ast::RangeLimits::Closed,
            })
        },
        hir::ExprKind::Struct(ref path, ref fields, None) => match path {
            hir::QPath::LangItem(hir::LangItem::RangeFull, _) => Some(Range {
                start: None,
                end: None,
                limits: ast::RangeLimits::HalfOpen,
            }),
            hir::QPath::LangItem(hir::LangItem::RangeFrom, _) => Some(Range {
                start: Some(get_field("start", fields)?),
                end: None,
                limits: ast::RangeLimits::HalfOpen,
            }),
            hir::QPath::LangItem(hir::LangItem::Range, _) => Some(Range {
                start: Some(get_field("start", fields)?),
                end: Some(get_field("end", fields)?),
                limits: ast::RangeLimits::HalfOpen,
            }),
            hir::QPath::LangItem(hir::LangItem::RangeToInclusive, _) => Some(Range {
                start: None,
                end: Some(get_field("end", fields)?),
                limits: ast::RangeLimits::Closed,
            }),
            hir::QPath::LangItem(hir::LangItem::RangeTo, _) => Some(Range {
                start: None,
                end: Some(get_field("end", fields)?),
                limits: ast::RangeLimits::HalfOpen,
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Checks if a `let` statement is from a `for` loop desugaring.
pub fn is_from_for_desugar(local: &hir::Local<'_>) -> bool {
    // This will detect plain for-loops without an actual variable binding:
    //
    // ```
    // for x in some_vec {
    //     // do stuff
    // }
    // ```
    if_chain! {
        if let Some(ref expr) = local.init;
        if let hir::ExprKind::Match(_, _, hir::MatchSource::ForLoopDesugar) = expr.kind;
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
pub fn for_loop<'tcx>(
    expr: &'tcx hir::Expr<'tcx>,
) -> Option<(&hir::Pat<'_>, &'tcx hir::Expr<'tcx>, &'tcx hir::Expr<'tcx>)> {
    if_chain! {
        if let hir::ExprKind::Match(ref iterexpr, ref arms, hir::MatchSource::ForLoopDesugar) = expr.kind;
        if let hir::ExprKind::Call(_, ref iterargs) = iterexpr.kind;
        if iterargs.len() == 1 && arms.len() == 1 && arms[0].guard.is_none();
        if let hir::ExprKind::Loop(ref block, _, _) = arms[0].body.kind;
        if block.expr.is_none();
        if let [ _, _, ref let_stmt, ref body ] = *block.stmts;
        if let hir::StmtKind::Local(ref local) = let_stmt.kind;
        if let hir::StmtKind::Expr(ref expr) = body.kind;
        then {
            return Some((&*local.pat, &iterargs[0], expr));
        }
    }
    None
}

/// Recover the essential nodes of a desugared while loop:
/// `while cond { body }` becomes `(cond, body)`.
pub fn while_loop<'tcx>(expr: &'tcx hir::Expr<'tcx>) -> Option<(&'tcx hir::Expr<'tcx>, &'tcx hir::Expr<'tcx>)> {
    if_chain! {
        if let hir::ExprKind::Loop(block, _, hir::LoopSource::While) = &expr.kind;
        if let hir::Block { expr: Some(expr), .. } = &**block;
        if let hir::ExprKind::Match(cond, arms, hir::MatchSource::WhileDesugar) = &expr.kind;
        if let hir::ExprKind::DropTemps(cond) = &cond.kind;
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
pub fn if_block<'tcx>(
    expr: &'tcx hir::Expr<'tcx>,
) -> Option<(
    &'tcx hir::Expr<'tcx>,
    &'tcx hir::Expr<'tcx>,
    Option<&'tcx hir::Expr<'tcx>>,
)> {
    if let hir::ExprKind::Match(ref cond, ref arms, hir::MatchSource::IfDesugar { contains_else_clause }) = expr.kind {
        let cond = if let hir::ExprKind::DropTemps(ref cond) = cond.kind {
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
    Repeat(&'a hir::Expr<'a>, &'a hir::Expr<'a>),
    /// `vec![a, b, c]`
    Vec(&'a [hir::Expr<'a>]),
}

/// Returns the arguments of the `vec!` macro if this expression was expanded
/// from `vec!`.
pub fn vec_macro<'e>(cx: &LateContext<'_>, expr: &'e hir::Expr<'_>) -> Option<VecArgs<'e>> {
    if_chain! {
        if let hir::ExprKind::Call(ref fun, ref args) = expr.kind;
        if let hir::ExprKind::Path(ref qpath) = fun.kind;
        if is_expn_of(fun.span, "vec").is_some();
        if let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id();
        then {
            return if match_def_path(cx, fun_def_id, &paths::VEC_FROM_ELEM) && args.len() == 2 {
                // `vec![elem; size]` case
                Some(VecArgs::Repeat(&args[0], &args[1]))
            }
            else if match_def_path(cx, fun_def_id, &paths::SLICE_INTO_VEC) && args.len() == 1 {
                // `vec![a, b, c]` case
                if_chain! {
                    if let hir::ExprKind::Box(ref boxed) = args[0].kind;
                    if let hir::ExprKind::Array(ref args) = boxed.kind;
                    then {
                        return Some(VecArgs::Vec(&*args));
                    }
                }

                None
            }
            else if match_def_path(cx, fun_def_id, &paths::VEC_NEW) && args.is_empty() {
                Some(VecArgs::Vec(&[]))
            }
            else {
                None
            };
        }
    }

    None
}

/// Extract args from an assert-like macro.
/// Currently working with:
/// - `assert!`, `assert_eq!` and `assert_ne!`
/// - `debug_assert!`, `debug_assert_eq!` and `debug_assert_ne!`
/// For example:
/// `assert!(expr)` will return Some([expr])
/// `debug_assert_eq!(a, b)` will return Some([a, b])
pub fn extract_assert_macro_args<'tcx>(e: &'tcx Expr<'tcx>) -> Option<Vec<&'tcx Expr<'tcx>>> {
    /// Try to match the AST for a pattern that contains a match, for example when two args are
    /// compared
    fn ast_matchblock(matchblock_expr: &'tcx Expr<'tcx>) -> Option<Vec<&Expr<'_>>> {
        if_chain! {
            if let ExprKind::Match(ref headerexpr, _, _) = &matchblock_expr.kind;
            if let ExprKind::Tup([lhs, rhs]) = &headerexpr.kind;
            if let ExprKind::AddrOf(BorrowKind::Ref, _, lhs) = lhs.kind;
            if let ExprKind::AddrOf(BorrowKind::Ref, _, rhs) = rhs.kind;
            then {
                return Some(vec![lhs, rhs]);
            }
        }
        None
    }

    if let ExprKind::Block(ref block, _) = e.kind {
        if block.stmts.len() == 1 {
            if let StmtKind::Semi(ref matchexpr) = block.stmts[0].kind {
                // macros with unique arg: `{debug_}assert!` (e.g., `debug_assert!(some_condition)`)
                if_chain! {
                    if let ExprKind::Match(ref ifclause, _, _) = matchexpr.kind;
                    if let ExprKind::DropTemps(ref droptmp) = ifclause.kind;
                    if let ExprKind::Unary(UnOp::UnNot, condition) = droptmp.kind;
                    then {
                        return Some(vec![condition]);
                    }
                }

                // debug macros with two args: `debug_assert_{ne, eq}` (e.g., `assert_ne!(a, b)`)
                if_chain! {
                    if let ExprKind::Block(ref matchblock,_) = matchexpr.kind;
                    if let Some(ref matchblock_expr) = matchblock.expr;
                    then {
                        return ast_matchblock(matchblock_expr);
                    }
                }
            }
        } else if let Some(matchblock_expr) = block.expr {
            // macros with two args: `assert_{ne, eq}` (e.g., `assert_ne!(a, b)`)
            return ast_matchblock(&matchblock_expr);
        }
    }
    None
}
