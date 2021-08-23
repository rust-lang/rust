//! This module contains functions that retrieves specifiec elements.

#![deny(clippy::missing_docs_in_private_items)]

use crate::{is_expn_of, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast::{self, LitKind};
use rustc_hir as hir;
use rustc_hir::{Block, BorrowKind, Expr, ExprKind, LoopSource, Node, Pat, StmtKind, UnOp};
use rustc_lint::LateContext;
use rustc_span::{sym, ExpnKind, Span, Symbol};

/// The essential nodes of a desugared for loop as well as the entire span:
/// `for pat in arg { body }` becomes `(pat, arg, body)`. Return `(pat, arg, body, span)`.
pub struct ForLoop<'tcx> {
    pub pat: &'tcx hir::Pat<'tcx>,
    pub arg: &'tcx hir::Expr<'tcx>,
    pub body: &'tcx hir::Expr<'tcx>,
    pub span: Span,
}

impl<'tcx> ForLoop<'tcx> {
    #[inline]
    pub fn hir(expr: &Expr<'tcx>) -> Option<Self> {
        if_chain! {
            if let hir::ExprKind::Match(ref iterexpr, ref arms, hir::MatchSource::ForLoopDesugar) = expr.kind;
            if let Some(first_arm) = arms.get(0);
            if let hir::ExprKind::Call(_, ref iterargs) = iterexpr.kind;
            if let Some(first_arg) = iterargs.get(0);
            if iterargs.len() == 1 && arms.len() == 1 && first_arm.guard.is_none();
            if let hir::ExprKind::Loop(ref block, ..) = first_arm.body.kind;
            if block.expr.is_none();
            if let [ _, _, ref let_stmt, ref body ] = *block.stmts;
            if let hir::StmtKind::Local(ref local) = let_stmt.kind;
            if let hir::StmtKind::Expr(ref body_expr) = body.kind;
            then {
                return Some(Self {
                    pat: &*local.pat,
                    arg: first_arg,
                    body: body_expr,
                    span: first_arm.span
                });
            }
        }
        None
    }
}

pub struct If<'hir> {
    pub cond: &'hir Expr<'hir>,
    pub r#else: Option<&'hir Expr<'hir>>,
    pub then: &'hir Expr<'hir>,
}

impl<'hir> If<'hir> {
    #[inline]
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
            Some(Self { cond, r#else, then })
        } else {
            None
        }
    }
}

pub struct IfLet<'hir> {
    pub let_pat: &'hir Pat<'hir>,
    pub let_expr: &'hir Expr<'hir>,
    pub if_then: &'hir Expr<'hir>,
    pub if_else: Option<&'hir Expr<'hir>>,
}

impl<'hir> IfLet<'hir> {
    pub fn hir(cx: &LateContext<'_>, expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::If(
            Expr {
                kind: ExprKind::Let(let_pat, let_expr, _),
                ..
            },
            if_then,
            if_else,
        ) = expr.kind
        {
            let hir = cx.tcx.hir();
            let mut iter = hir.parent_iter(expr.hir_id);
            if let Some((_, Node::Block(Block { stmts: [], .. }))) = iter.next() {
                if let Some((_, Node::Expr(Expr { kind: ExprKind::Loop(_, _, LoopSource::While, _), .. }))) = iter.next() {
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

pub struct IfOrIfLet<'hir> {
    pub cond: &'hir Expr<'hir>,
    pub r#else: Option<&'hir Expr<'hir>>,
    pub then: &'hir Expr<'hir>,
}

impl<'hir> IfOrIfLet<'hir> {
    #[inline]
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
                return Some(Self { cond, r#else, then });
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
}

/// Represent the pre-expansion arguments of a `vec!` invocation.
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
}

pub struct While<'hir> {
    pub if_cond: &'hir Expr<'hir>,
    pub if_then: &'hir Expr<'hir>,
    pub if_else: Option<&'hir Expr<'hir>>,
}

impl<'hir> While<'hir> {
    #[inline]
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::Loop(
            Block {
                expr:
                    Some(Expr {
                        kind:
                            ExprKind::If(
                                Expr {
                                    kind: ExprKind::DropTemps(if_cond),
                                    ..
                                },
                                if_then,
                                if_else_ref,
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
            let if_else = *if_else_ref;
            return Some(Self {
                if_cond,
                if_then,
                if_else,
            });
        }
        None
    }
}

pub struct WhileLet<'hir> {
    pub if_expr: &'hir Expr<'hir>,
    pub let_pat: &'hir Pat<'hir>,
    pub let_expr: &'hir Expr<'hir>,
    pub if_then: &'hir Expr<'hir>,
    pub if_else: Option<&'hir Expr<'hir>>,
}

impl<'hir> WhileLet<'hir> {
    #[inline]
    pub const fn hir(expr: &Expr<'hir>) -> Option<Self> {
        if let ExprKind::Loop(
            Block {
                expr: Some(if_expr), ..
            },
            _,
            LoopSource::While,
            _,
        ) = expr.kind
        {
            if let Expr {
                kind:
                    ExprKind::If(
                        Expr {
                            kind: ExprKind::Let(let_pat, let_expr, _),
                            ..
                        },
                        if_then,
                        if_else_ref,
                    ),
                ..
            } = if_expr
            {
                let if_else = *if_else_ref;
                return Some(Self {
                    if_expr,
                    let_pat,
                    let_expr,
                    if_then,
                    if_else,
                });
            }
        }
        None
    }
}

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

/// Extract args from an assert-like macro.
/// Currently working with:
/// - `assert!`, `assert_eq!` and `assert_ne!`
/// - `debug_assert!`, `debug_assert_eq!` and `debug_assert_ne!`
/// For example:
/// `assert!(expr)` will return `Some([expr])`
/// `debug_assert_eq!(a, b)` will return `Some([a, b])`
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
                    if let Some(If { cond, .. }) = If::hir(matchexpr);
                    if let ExprKind::Unary(UnOp::Not, condition) = cond.kind;
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
    /// Symbols corresponding to [`Self::format_string_parts`]
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

            if let ExprKind::Match(inner_match, [arm], _) = expr.kind;

            // `match match`, if you will
            if let ExprKind::Match(args, [inner_arm], _) = inner_match.kind;
            if let ExprKind::Tup(value_args) = args.kind;
            if let Some(value_args) = value_args
                .iter()
                .map(|e| match e.kind {
                    ExprKind::AddrOf(_, _, e) => Some(e),
                    _ => None,
                })
                .collect();
            if let ExprKind::Array(args) = inner_arm.body.kind;

            if let ExprKind::Block(Block { stmts: [], expr: Some(expr), .. }, _) = arm.body.kind;
            if let ExprKind::Call(_, call_args) = expr.kind;
            if let Some((strs_ref, fmt_expr)) = match call_args {
                // Arguments::new_v1
                [strs_ref, _] => Some((strs_ref, None)),
                // Arguments::new_v1_formatted
                [strs_ref, _, fmt_expr] => Some((strs_ref, Some(fmt_expr))),
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
