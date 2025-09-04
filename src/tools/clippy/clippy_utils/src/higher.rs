//! This module contains functions that retrieve specific elements.

#![deny(clippy::missing_docs_in_private_items)]

use crate::consts::{ConstEvalCtxt, Constant};
use crate::ty::is_type_diagnostic_item;
use crate::{is_expn_of, sym};

use rustc_ast::ast;
use rustc_hir as hir;
use rustc_hir::{Arm, Block, Expr, ExprKind, HirId, LoopSource, MatchSource, Node, Pat, QPath, StructTailExpr};
use rustc_lint::LateContext;
use rustc_span::{Span, symbol};

/// The essential nodes of a desugared for loop as well as the entire span:
/// `for pat in arg { body }` becomes `(pat, arg, body)`. Returns `(pat, arg, body, span)`.
pub struct ForLoop<'tcx> {
    /// `for` loop item
    pub pat: &'tcx Pat<'tcx>,
    /// `IntoIterator` argument
    pub arg: &'tcx Expr<'tcx>,
    /// `for` loop body
    pub body: &'tcx Expr<'tcx>,
    /// Compare this against `hir::Destination.target`
    pub loop_id: HirId,
    /// entire `for` loop span
    pub span: Span,
    /// label
    pub label: Option<ast::Label>,
}

impl<'tcx> ForLoop<'tcx> {
    /// Parses a desugared `for` loop
    pub fn hir(expr: &Expr<'tcx>) -> Option<Self> {
        if let ExprKind::DropTemps(e) = expr.kind
            && let ExprKind::Match(iterexpr, [arm], MatchSource::ForLoopDesugar) = e.kind
            && let ExprKind::Call(_, [arg]) = iterexpr.kind
            && let ExprKind::Loop(block, label, ..) = arm.body.kind
            && let [stmt] = block.stmts
            && let hir::StmtKind::Expr(e) = stmt.kind
            && let ExprKind::Match(_, [_, some_arm], _) = e.kind
            && let hir::PatKind::Struct(_, [field], _) = some_arm.pat.kind
        {
            return Some(Self {
                pat: field.pat,
                arg,
                body: some_arm.body,
                loop_id: arm.body.hir_id,
                span: expr.span.ctxt().outer_expn_data().call_site,
                label,
            });
        }
        None
    }
}

/// An `if` expression without `let`
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
    /// Parses an `if` expression without `let`
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::If(cond, then, r#else) = expr.kind
            && !has_let_expr(cond)
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
    /// `if let PAT = EXPR`
    ///     ^^^^^^^^^^^^^^
    pub let_span: Span,
}

impl<'hir> IfLet<'hir> {
    /// Parses an `if let` expression
    pub fn hir(cx: &LateContext<'_>, expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::If(
            &Expr {
                kind:
                    ExprKind::Let(&hir::LetExpr {
                        pat: let_pat,
                        init: let_expr,
                        span: let_span,
                        ..
                    }),
                ..
            },
            if_then,
            if_else,
        ) = expr.kind
        {
            let mut iter = cx.tcx.hir_parent_iter(expr.hir_id);
            if let Some((_, Node::Block(Block { stmts: [], .. }))) = iter.next()
                && let Some((
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
            return Some(Self {
                let_pat,
                let_expr,
                if_then,
                if_else,
                let_span,
            });
        }
        None
    }
}

/// An `if let` or `match` expression. Useful for lints that trigger on one or the other.
#[derive(Debug)]
pub enum IfLetOrMatch<'hir> {
    /// Any `match` expression
    Match(&'hir Expr<'hir>, &'hir [Arm<'hir>], MatchSource),
    /// scrutinee, pattern, then block, else block
    IfLet(
        &'hir Expr<'hir>,
        &'hir Pat<'hir>,
        &'hir Expr<'hir>,
        Option<&'hir Expr<'hir>>,
        /// `if let PAT = EXPR`
        ///     ^^^^^^^^^^^^^^
        Span,
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
                     let_span,
                 }| { Self::IfLet(let_expr, let_pat, if_then, if_else, let_span) },
            ),
        }
    }

    pub fn scrutinee(&self) -> &'hir Expr<'hir> {
        match self {
            Self::Match(scrutinee, _, _) | Self::IfLet(scrutinee, _, _, _, _) => scrutinee,
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
            Some(Self { cond, then, r#else })
        } else {
            None
        }
    }
}

/// Represent a range akin to `ast::ExprKind::Range`.
#[derive(Debug, Copy, Clone)]
pub struct Range<'a> {
    /// The lower bound of the range, or `None` for ranges such as `..X`.
    pub start: Option<&'a Expr<'a>>,
    /// The upper bound of the range, or `None` for ranges such as `X..`.
    pub end: Option<&'a Expr<'a>>,
    /// Whether the interval is open or closed.
    pub limits: ast::RangeLimits,
}

impl<'a> Range<'a> {
    /// Higher a `hir` range to something similar to `ast::ExprKind::Range`.
    #[allow(clippy::similar_names)]
    pub fn hir(expr: &'a Expr<'_>) -> Option<Range<'a>> {
        match expr.kind {
            ExprKind::Call(path, [arg1, arg2])
                if matches!(
                    path.kind,
                    ExprKind::Path(QPath::LangItem(hir::LangItem::RangeInclusiveNew, ..))
                ) =>
            {
                Some(Range {
                    start: Some(arg1),
                    end: Some(arg2),
                    limits: ast::RangeLimits::Closed,
                })
            },
            ExprKind::Struct(path, fields, StructTailExpr::None) => match (path, fields) {
                (QPath::LangItem(hir::LangItem::RangeFull, ..), []) => Some(Range {
                    start: None,
                    end: None,
                    limits: ast::RangeLimits::HalfOpen,
                }),
                (QPath::LangItem(hir::LangItem::RangeFrom, ..), [field]) if field.ident.name == sym::start => {
                    Some(Range {
                        start: Some(field.expr),
                        end: None,
                        limits: ast::RangeLimits::HalfOpen,
                    })
                },
                (QPath::LangItem(hir::LangItem::Range, ..), [field1, field2]) => {
                    let (start, end) = match (field1.ident.name, field2.ident.name) {
                        (sym::start, sym::end) => (field1.expr, field2.expr),
                        (sym::end, sym::start) => (field2.expr, field1.expr),
                        _ => return None,
                    };
                    Some(Range {
                        start: Some(start),
                        end: Some(end),
                        limits: ast::RangeLimits::HalfOpen,
                    })
                },
                (QPath::LangItem(hir::LangItem::RangeToInclusive, ..), [field]) if field.ident.name == sym::end => {
                    Some(Range {
                        start: None,
                        end: Some(field.expr),
                        limits: ast::RangeLimits::Closed,
                    })
                },
                (QPath::LangItem(hir::LangItem::RangeTo, ..), [field]) if field.ident.name == sym::end => Some(Range {
                    start: None,
                    end: Some(field.expr),
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
    Repeat(&'a Expr<'a>, &'a Expr<'a>),
    /// `vec![a, b, c]`
    Vec(&'a [Expr<'a>]),
}

impl<'a> VecArgs<'a> {
    /// Returns the arguments of the `vec!` macro if this expression was expanded
    /// from `vec!`.
    pub fn hir(cx: &LateContext<'_>, expr: &'a Expr<'_>) -> Option<VecArgs<'a>> {
        if let ExprKind::Call(fun, args) = expr.kind
            && let ExprKind::Path(ref qpath) = fun.kind
            && is_expn_of(fun.span, sym::vec).is_some()
            && let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id()
            && let Some(name) = cx.tcx.get_diagnostic_name(fun_def_id)
        {
            return match (name, args) {
                (sym::vec_from_elem, [elem, size]) => {
                    // `vec![elem; size]` case
                    Some(VecArgs::Repeat(elem, size))
                },
                (sym::slice_into_vec, [slice])
                    if let ExprKind::Call(_, [arg]) = slice.kind
                        && let ExprKind::Array(args) = arg.kind =>
                {
                    // `vec![a, b, c]` case
                    Some(VecArgs::Vec(args))
                },
                (sym::vec_new, []) => Some(VecArgs::Vec(&[])),
                _ => None,
            };
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
                        kind: ExprKind::If(condition, body, _),
                        ..
                    }),
                ..
            },
            _,
            LoopSource::While,
            span,
        ) = expr.kind
            && !has_let_expr(condition)
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
    pub label: Option<ast::Label>,
    /// `while let PAT = EXPR`
    ///        ^^^^^^^^^^^^^^
    pub let_span: Span,
}

impl<'hir> WhileLet<'hir> {
    #[inline]
    /// Parses a desugared `while let` loop
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::Loop(
            &Block {
                expr:
                    Some(&Expr {
                        kind:
                            ExprKind::If(
                                &Expr {
                                    kind:
                                        ExprKind::Let(&hir::LetExpr {
                                            pat: let_pat,
                                            init: let_expr,
                                            span: let_span,
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
            label,
            LoopSource::While,
            _,
        ) = expr.kind
        {
            return Some(Self {
                let_pat,
                let_expr,
                if_then,
                label,
                let_span,
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
                } else if name.ident.name == sym::with_capacity {
                    let arg = args.first()?;
                    return match ConstEvalCtxt::new(cx).eval_simple(arg) {
                        Some(Constant::Int(num)) => Some(VecInitKind::WithConstCapacity(num)),
                        _ => Some(VecInitKind::WithExprCapacity(arg.hir_id)),
                    };
                }
            },
            ExprKind::Path(QPath::Resolved(_, path))
                if cx.tcx.is_diagnostic_item(sym::default_fn, path.res.opt_def_id()?)
                    && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::Vec) =>
            {
                return Some(VecInitKind::Default);
            },
            _ => (),
        }
    }
    None
}

/// Checks that a condition doesn't have a `let` expression, to keep `If` and `While` from accepting
/// `if let` and `while let`.
pub const fn has_let_expr<'tcx>(cond: &'tcx Expr<'tcx>) -> bool {
    match &cond.kind {
        ExprKind::Let(_) => true,
        ExprKind::Binary(_, lhs, rhs) => has_let_expr(lhs) || has_let_expr(rhs),
        _ => false,
    }
}
