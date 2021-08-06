//! This module contains functions for retrieve the original AST from lowered
//! `hir`.

#![deny(clippy::missing_docs_in_private_items)]

use crate::{is_expn_of, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast::{self, LitKind};
use rustc_hir as hir;
use rustc_hir::{BorrowKind, Expr, ExprKind, StmtKind, UnOp};
use rustc_lint::LateContext;
use rustc_span::{sym, ExpnKind, Span, Symbol};

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
    fn get_field<'c>(name: &str, fields: &'c [hir::ExprField<'_>]) -> Option<&'c hir::Expr<'c>> {
        let expr = &fields.iter().find(|field| field.ident.name.as_str() == name)?.expr;

        Some(expr)
    }

    match expr.kind {
        hir::ExprKind::Call(path, args)
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
        hir::ExprKind::Struct(path, fields, None) => match path {
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
        if let Some(expr) = local.init;
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

/// Recover the essential nodes of a desugared for loop as well as the entire span:
/// `for pat in arg { body }` becomes `(pat, arg, body)`. Return `(pat, arg, body, span)`.
pub fn for_loop<'tcx>(
    expr: &'tcx hir::Expr<'tcx>,
) -> Option<(&hir::Pat<'_>, &'tcx hir::Expr<'tcx>, &'tcx hir::Expr<'tcx>, Span)> {
    if_chain! {
        if let hir::ExprKind::Match(iterexpr, arms, hir::MatchSource::ForLoopDesugar) = expr.kind;
        if let hir::ExprKind::Call(_, iterargs) = iterexpr.kind;
        if iterargs.len() == 1 && arms.len() == 1 && arms[0].guard.is_none();
        if let hir::ExprKind::Loop(block, ..) = arms[0].body.kind;
        if block.expr.is_none();
        if let [ _, _, ref let_stmt, ref body ] = *block.stmts;
        if let hir::StmtKind::Local(local) = let_stmt.kind;
        if let hir::StmtKind::Expr(expr) = body.kind;
        then {
            return Some((&*local.pat, &iterargs[0], expr, arms[0].span));
        }
    }
    None
}

/// Recover the essential nodes of a desugared while loop:
/// `while cond { body }` becomes `(cond, body)`.
pub fn while_loop<'tcx>(expr: &'tcx hir::Expr<'tcx>) -> Option<(&'tcx hir::Expr<'tcx>, &'tcx hir::Expr<'tcx>)> {
    if_chain! {
        if let hir::ExprKind::Loop(hir::Block { expr: Some(expr), .. }, _, hir::LoopSource::While, _) = &expr.kind;
        if let hir::ExprKind::Match(cond, arms, hir::MatchSource::WhileDesugar) = &expr.kind;
        if let hir::ExprKind::DropTemps(cond) = &cond.kind;
        if let [hir::Arm { body, .. }, ..] = &arms[..];
        then {
            return Some((cond, body));
        }
    }
    None
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
        if let hir::ExprKind::Call(fun, args) = expr.kind;
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
                    if let hir::ExprKind::Box(boxed) = args[0].kind;
                    if let hir::ExprKind::Array(args) = boxed.kind;
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
            if let ExprKind::Match(headerexpr, _, _) = &matchblock_expr.kind;
            if let ExprKind::Tup([lhs, rhs]) = &headerexpr.kind;
            if let ExprKind::AddrOf(BorrowKind::Ref, _, lhs) = lhs.kind;
            if let ExprKind::AddrOf(BorrowKind::Ref, _, rhs) = rhs.kind;
            then {
                return Some(vec![lhs, rhs]);
            }
        }
        None
    }

    if let ExprKind::Block(block, _) = e.kind {
        if block.stmts.len() == 1 {
            if let StmtKind::Semi(matchexpr) = block.stmts.get(0)?.kind {
                // macros with unique arg: `{debug_}assert!` (e.g., `debug_assert!(some_condition)`)
                if_chain! {
                    if let ExprKind::If(clause, _, _)  = matchexpr.kind;
                    if let ExprKind::Unary(UnOp::Not, condition) = clause.kind;
                    then {
                        return Some(vec![condition]);
                    }
                }

                // debug macros with two args: `debug_assert_{ne, eq}` (e.g., `assert_ne!(a, b)`)
                if_chain! {
                    if let ExprKind::Block(matchblock,_) = matchexpr.kind;
                    if let Some(matchblock_expr) = matchblock.expr;
                    then {
                        return ast_matchblock(matchblock_expr);
                    }
                }
            }
        } else if let Some(matchblock_expr) = block.expr {
            // macros with two args: `assert_{ne, eq}` (e.g., `assert_ne!(a, b)`)
            return ast_matchblock(matchblock_expr);
        }
    }
    None
}

/// A parsed `format!` expansion
pub struct FormatExpn<'tcx> {
    /// Span of `format!(..)`
    pub call_site: Span,
    /// Inner `format_args!` expansion
    pub format_args: FormatArgsExpn<'tcx>,
}

impl FormatExpn<'tcx> {
    /// Parses an expanded `format!` invocation
    pub fn parse(expr: &'tcx Expr<'tcx>) -> Option<Self> {
        if_chain! {
            if let ExprKind::Block(block, _) = expr.kind;
            if let [stmt] = block.stmts;
            if let StmtKind::Local(local) = stmt.kind;
            if let Some(init) = local.init;
            if let ExprKind::Call(_, [format_args]) = init.kind;
            let expn_data = expr.span.ctxt().outer_expn_data();
            if let ExpnKind::Macro(_, sym::format) = expn_data.kind;
            if let Some(format_args) = FormatArgsExpn::parse(format_args);
            then {
                Some(FormatExpn {
                    call_site: expn_data.call_site,
                    format_args,
                })
            } else {
                None
            }
        }
    }
}

/// A parsed `format_args!` expansion
pub struct FormatArgsExpn<'tcx> {
    /// Span of the first argument, the format string
    pub format_string_span: Span,
    /// Values passed after the format string
    pub value_args: Vec<&'tcx Expr<'tcx>>,

    /// String literal expressions which represent the format string split by "{}"
    pub format_string_parts: &'tcx [Expr<'tcx>],
    /// Symbols corresponding to [`format_string_parts`]
    pub format_string_symbols: Vec<Symbol>,
    /// Expressions like `ArgumentV1::new(arg0, Debug::fmt)`
    pub args: &'tcx [Expr<'tcx>],
    /// The final argument passed to `Arguments::new_v1_formatted`, if applicable
    pub fmt_expr: Option<&'tcx Expr<'tcx>>,
}

impl FormatArgsExpn<'tcx> {
    /// Parses an expanded `format_args!` or `format_args_nl!` invocation
    pub fn parse(expr: &'tcx Expr<'tcx>) -> Option<Self> {
        if_chain! {
            if let ExpnKind::Macro(_, name) = expr.span.ctxt().outer_expn_data().kind;
            let name = name.as_str();
            if name.ends_with("format_args") || name.ends_with("format_args_nl");
            if let ExprKind::Call(_, args) = expr.kind;
            if let Some((strs_ref, args, fmt_expr)) = match args {
                // Arguments::new_v1
                [strs_ref, args] => Some((strs_ref, args, None)),
                // Arguments::new_v1_formatted
                [strs_ref, args, fmt_expr] => Some((strs_ref, args, Some(fmt_expr))),
                _ => None,
            };
            if let ExprKind::AddrOf(BorrowKind::Ref, _, strs_arr) = strs_ref.kind;
            if let ExprKind::Array(format_string_parts) = strs_arr.kind;
            if let Some(format_string_symbols) = format_string_parts
                .iter()
                .map(|e| {
                    if let ExprKind::Lit(lit) = &e.kind {
                        if let LitKind::Str(symbol, _style) = lit.node {
                            return Some(symbol);
                        }
                    }
                    None
                })
                .collect();
            if let ExprKind::AddrOf(BorrowKind::Ref, _, args) = args.kind;
            if let ExprKind::Match(args, [arm], _) = args.kind;
            if let ExprKind::Tup(value_args) = args.kind;
            if let Some(value_args) = value_args
                .iter()
                .map(|e| match e.kind {
                    ExprKind::AddrOf(_, _, e) => Some(e),
                    _ => None,
                })
                .collect();
            if let ExprKind::Array(args) = arm.body.kind;
            then {
                Some(FormatArgsExpn {
                    format_string_span: strs_ref.span,
                    value_args,
                    format_string_parts,
                    format_string_symbols,
                    args,
                    fmt_expr,
                })
            } else {
                None
            }
        }
    }
}
