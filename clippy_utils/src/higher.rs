//! This module contains functions that retrieve specific elements.

#![deny(clippy::missing_docs_in_private_items)]

use crate::consts::{constant_simple, Constant};
use crate::ty::is_type_diagnostic_item;
use crate::{is_expn_of, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_hir as hir;
use rustc_hir::{Arm, Block, Expr, ExprKind, HirId, LoopSource, MatchSource, Node, Pat, QPath};
use rustc_lint::LateContext;
use rustc_span::{sym, symbol, Span};

/// The essential nodes of a desugared for loop as well as the entire span:
/// `for pat in arg { body }` becomes `(pat, arg, body)`. Returns `(pat, arg, body, span)`.
pub struct ForLoop<'tcx> {
    /// `for` loop item
    pub pat: &'tcx hir::Pat<'tcx>,
    /// `IntoIterator` argument
    pub arg: &'tcx hir::Expr<'tcx>,
    /// `for` loop body
    pub body: &'tcx hir::Expr<'tcx>,
    /// Compare this against `hir::Destination.target`
    pub loop_id: HirId,
    /// entire `for` loop span
    pub span: Span,
}

impl<'tcx> ForLoop<'tcx> {
    /// Parses a desugared `for` loop
    pub fn hir(expr: &Expr<'tcx>) -> Option<Self> {
        if_chain! {
            if let hir::ExprKind::DropTemps(e) = expr.kind;
            if let hir::ExprKind::Match(iterexpr, [arm], hir::MatchSource::ForLoopDesugar) = e.kind;
            if let hir::ExprKind::Call(_, [arg]) = iterexpr.kind;
            if let hir::ExprKind::Loop(block, ..) = arm.body.kind;
            if let [stmt] = block.stmts;
            if let hir::StmtKind::Expr(e) = stmt.kind;
            if let hir::ExprKind::Match(_, [_, some_arm], _) = e.kind;
            if let hir::PatKind::Struct(_, [field], _) = some_arm.pat.kind;
            then {
                return Some(Self {
                    pat: field.pat,
                    arg,
                    body: some_arm.body,
                    loop_id: arm.body.hir_id,
                    span: expr.span.ctxt().outer_expn_data().call_site,
                });
            }
        }
        None
    }
}

/// An `if` expression without `DropTemps`
pub struct If<'hir> {
    /// `if` condition
    pub cond: &'hir Expr<'hir>,
    /// `if` then expression
    pub then: &'hir Expr<'hir>,
    /// `else` expression
    pub r#else: Option<&'hir Expr<'hir>>,
}

impl<'hir> If<'hir> {
    #[inline]
    /// Parses an `if` expression
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::If(
            Expr {
                kind: ExprKind::DropTemps(cond),
                ..
            },
            then,
            r#else,
        ) = expr.kind
        {
            Some(Self { cond, then, r#else })
        } else {
            None
        }
    }
}

/// An `if let` expression
pub struct IfLet<'hir> {
    /// `if let` pattern
    pub let_pat: &'hir Pat<'hir>,
    /// `if let` scrutinee
    pub let_expr: &'hir Expr<'hir>,
    /// `if let` then expression
    pub if_then: &'hir Expr<'hir>,
    /// `if let` else expression
    pub if_else: Option<&'hir Expr<'hir>>,
}

impl<'hir> IfLet<'hir> {
    /// Parses an `if let` expression
    pub fn hir(cx: &LateContext<'_>, expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::If(
            Expr {
                kind:
                    ExprKind::Let(hir::Let {
                        pat: let_pat,
                        init: let_expr,
                        ..
                    }),
                ..
            },
            if_then,
            if_else,
        ) = expr.kind
        {
            let mut iter = cx.tcx.hir().parent_iter(expr.hir_id);
            if let Some((_, Node::Block(Block { stmts: [], .. }))) = iter.next() {
                if let Some((
                    _,
                    Node::Expr(Expr {
                        kind: ExprKind::Loop(_, _, LoopSource::While, _),
                        ..
                    }),
                )) = iter.next()
                {
                    // while loop desugar
                    return None;
                }
            }
            return Some(Self {
                let_pat,
                let_expr,
                if_then,
                if_else,
            });
        }
        None
    }
}

/// An `if let` or `match` expression. Useful for lints that trigger on one or the other.
pub enum IfLetOrMatch<'hir> {
    /// Any `match` expression
    Match(&'hir Expr<'hir>, &'hir [Arm<'hir>], MatchSource),
    /// scrutinee, pattern, then block, else block
    IfLet(
        &'hir Expr<'hir>,
        &'hir Pat<'hir>,
        &'hir Expr<'hir>,
        Option<&'hir Expr<'hir>>,
    ),
}

impl<'hir> IfLetOrMatch<'hir> {
    /// Parses an `if let` or `match` expression
    pub fn parse(cx: &LateContext<'_>, expr: &Expr<'hir>) -> Option<Self> {
        match expr.kind {
            ExprKind::Match(expr, arms, source) => Some(Self::Match(expr, arms, source)),
            _ => IfLet::hir(cx, expr).map(
                |IfLet {
                     let_expr,
                     let_pat,
                     if_then,
                     if_else,
                 }| { Self::IfLet(let_expr, let_pat, if_then, if_else) },
            ),
        }
    }
}

/// An `if` or `if let` expression
pub struct IfOrIfLet<'hir> {
    /// `if` condition that is maybe a `let` expression
    pub cond: &'hir Expr<'hir>,
    /// `if` then expression
    pub then: &'hir Expr<'hir>,
    /// `else` expression
    pub r#else: Option<&'hir Expr<'hir>>,
}

impl<'hir> IfOrIfLet<'hir> {
    #[inline]
    /// Parses an `if` or `if let` expression
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::If(cond, then, r#else) = expr.kind {
            if let ExprKind::DropTemps(new_cond) = cond.kind {
                return Some(Self {
                    cond: new_cond,
                    r#else,
                    then,
                });
            }
            if let ExprKind::Let(..) = cond.kind {
                return Some(Self { cond, then, r#else });
            }
        }
        None
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

impl<'a> Range<'a> {
    /// Higher a `hir` range to something similar to `ast::ExprKind::Range`.
    pub fn hir(expr: &'a hir::Expr<'_>) -> Option<Range<'a>> {
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
                    hir::ExprKind::Path(hir::QPath::LangItem(hir::LangItem::RangeInclusiveNew, ..))
                ) =>
            {
                Some(Range {
                    start: Some(&args[0]),
                    end: Some(&args[1]),
                    limits: ast::RangeLimits::Closed,
                })
            },
            hir::ExprKind::Struct(path, fields, None) => match &path {
                hir::QPath::LangItem(hir::LangItem::RangeFull, ..) => Some(Range {
                    start: None,
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                }),
                hir::QPath::LangItem(hir::LangItem::RangeFrom, ..) => Some(Range {
                    start: Some(get_field("start", fields)?),
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                }),
                hir::QPath::LangItem(hir::LangItem::Range, ..) => Some(Range {
                    start: Some(get_field("start", fields)?),
                    end: Some(get_field("end", fields)?),
                    limits: ast::RangeLimits::HalfOpen,
                }),
                hir::QPath::LangItem(hir::LangItem::RangeToInclusive, ..) => Some(Range {
                    start: None,
                    end: Some(get_field("end", fields)?),
                    limits: ast::RangeLimits::Closed,
                }),
                hir::QPath::LangItem(hir::LangItem::RangeTo, ..) => Some(Range {
                    start: None,
                    end: Some(get_field("end", fields)?),
                    limits: ast::RangeLimits::HalfOpen,
                }),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Represents the pre-expansion arguments of a `vec!` invocation.
pub enum VecArgs<'a> {
    /// `vec![elem; len]`
    Repeat(&'a hir::Expr<'a>, &'a hir::Expr<'a>),
    /// `vec![a, b, c]`
    Vec(&'a [hir::Expr<'a>]),
}

impl<'a> VecArgs<'a> {
    /// Returns the arguments of the `vec!` macro if this expression was expanded
    /// from `vec!`.
    pub fn hir(cx: &LateContext<'_>, expr: &'a hir::Expr<'_>) -> Option<VecArgs<'a>> {
        if_chain! {
            if let hir::ExprKind::Call(fun, args) = expr.kind;
            if let hir::ExprKind::Path(ref qpath) = fun.kind;
            if is_expn_of(fun.span, "vec").is_some();
            if let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id();
            then {
                return if match_def_path(cx, fun_def_id, &paths::VEC_FROM_ELEM) && args.len() == 2 {
                    // `vec![elem; size]` case
                    Some(VecArgs::Repeat(&args[0], &args[1]))
                } else if match_def_path(cx, fun_def_id, &paths::SLICE_INTO_VEC) && args.len() == 1 {
                    // `vec![a, b, c]` case
                    if let hir::ExprKind::Call(_, [arg]) = &args[0].kind
                        && let hir::ExprKind::Array(args) = arg.kind {
                        Some(VecArgs::Vec(args))
                    } else {
                        None
                    }
                } else if match_def_path(cx, fun_def_id, &paths::VEC_NEW) && args.is_empty() {
                    Some(VecArgs::Vec(&[]))
                } else {
                    None
                };
            }
        }

        None
    }
}

/// A desugared `while` loop
pub struct While<'hir> {
    /// `while` loop condition
    pub condition: &'hir Expr<'hir>,
    /// `while` loop body
    pub body: &'hir Expr<'hir>,
    /// Span of the loop header
    pub span: Span,
}

impl<'hir> While<'hir> {
    #[inline]
    /// Parses a desugared `while` loop
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::Loop(
            Block {
                expr:
                    Some(Expr {
                        kind:
                            ExprKind::If(
                                Expr {
                                    kind: ExprKind::DropTemps(condition),
                                    ..
                                },
                                body,
                                _,
                            ),
                        ..
                    }),
                ..
            },
            _,
            LoopSource::While,
            span,
        ) = expr.kind
        {
            return Some(Self { condition, body, span });
        }
        None
    }
}

/// A desugared `while let` loop
pub struct WhileLet<'hir> {
    /// `while let` loop item pattern
    pub let_pat: &'hir Pat<'hir>,
    /// `while let` loop scrutinee
    pub let_expr: &'hir Expr<'hir>,
    /// `while let` loop body
    pub if_then: &'hir Expr<'hir>,
}

impl<'hir> WhileLet<'hir> {
    #[inline]
    /// Parses a desugared `while let` loop
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::Loop(
            Block {
                expr:
                    Some(Expr {
                        kind:
                            ExprKind::If(
                                Expr {
                                    kind:
                                        ExprKind::Let(hir::Let {
                                            pat: let_pat,
                                            init: let_expr,
                                            ..
                                        }),
                                    ..
                                },
                                if_then,
                                _,
                            ),
                        ..
                    }),
                ..
            },
            _,
            LoopSource::While,
            _,
        ) = expr.kind
        {
            return Some(Self {
                let_pat,
                let_expr,
                if_then,
            });
        }
        None
    }
}

/// Converts a `hir` binary operator to the corresponding `ast` type.
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

/// A parsed `Vec` initialization expression
#[derive(Clone, Copy)]
pub enum VecInitKind {
    /// `Vec::new()`
    New,
    /// `Vec::default()` or `Default::default()`
    Default,
    /// `Vec::with_capacity(123)`
    WithConstCapacity(u128),
    /// `Vec::with_capacity(slice.len())`
    WithExprCapacity(HirId),
}

/// Checks if the given expression is an initialization of `Vec` and returns its kind.
pub fn get_vec_init_kind<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<VecInitKind> {
    if let ExprKind::Call(func, args) = expr.kind {
        match func.kind {
            ExprKind::Path(QPath::TypeRelative(ty, name))
                if is_type_diagnostic_item(cx, cx.typeck_results().node_type(ty.hir_id), sym::Vec) =>
            {
                if name.ident.name == sym::new {
                    return Some(VecInitKind::New);
                } else if name.ident.name == symbol::kw::Default {
                    return Some(VecInitKind::Default);
                } else if name.ident.name.as_str() == "with_capacity" {
                    let arg = args.get(0)?;
                    return match constant_simple(cx, cx.typeck_results(), arg) {
                        Some(Constant::Int(num)) => Some(VecInitKind::WithConstCapacity(num)),
                        _ => Some(VecInitKind::WithExprCapacity(arg.hir_id)),
                    };
                };
            },
            ExprKind::Path(QPath::Resolved(_, path))
                if match_def_path(cx, path.res.opt_def_id()?, &paths::DEFAULT_TRAIT_METHOD)
                    && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::Vec) =>
            {
                return Some(VecInitKind::Default);
            },
            _ => (),
        }
    }
    None
}
