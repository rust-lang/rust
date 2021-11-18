//! This module contains functions that retrieve specific elements.

#![deny(clippy::missing_docs_in_private_items)]

use crate::ty::is_type_diagnostic_item;
use crate::{is_expn_of, last_path_segment, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast::{self, LitKind};
use rustc_hir as hir;
use rustc_hir::{
    Arm, Block, BorrowKind, Expr, ExprKind, HirId, LoopSource, MatchSource, Node, Pat, QPath, StmtKind, UnOp,
};
use rustc_lint::LateContext;
use rustc_span::{sym, symbol, ExpnKind, Span, Symbol};

/// The essential nodes of a desugared for loop as well as the entire span:
/// `for pat in arg { body }` becomes `(pat, arg, body)`. Return `(pat, arg, body, span)`.
pub struct ForLoop<'tcx> {
    /// `for` loop item
    pub pat: &'tcx hir::Pat<'tcx>,
    /// `IntoIterator` argument
    pub arg: &'tcx hir::Expr<'tcx>,
    /// `for` loop body
    pub body: &'tcx hir::Expr<'tcx>,
    /// entire `for` loop span
    pub span: Span,
}

impl<'tcx> ForLoop<'tcx> {
    #[inline]
    /// Parses a desugared `for` loop
    pub fn hir(expr: &Expr<'tcx>) -> Option<Self> {
        if_chain! {
            if let hir::ExprKind::Match(iterexpr, arms, hir::MatchSource::ForLoopDesugar) = expr.kind;
            if let Some(first_arm) = arms.get(0);
            if let hir::ExprKind::Call(_, iterargs) = iterexpr.kind;
            if let Some(first_arg) = iterargs.get(0);
            if iterargs.len() == 1 && arms.len() == 1 && first_arm.guard.is_none();
            if let hir::ExprKind::Loop(block, ..) = first_arm.body.kind;
            if block.expr.is_none();
            if let [ _, _, ref let_stmt, ref body ] = *block.stmts;
            if let hir::StmtKind::Local(local) = let_stmt.kind;
            if let hir::StmtKind::Expr(body_expr) = body.kind;
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
                kind: ExprKind::Let(let_pat, let_expr, _),
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
                    hir::ExprKind::Path(hir::QPath::LangItem(hir::LangItem::RangeInclusiveNew, _))
                ) =>
            {
                Some(Range {
                    start: Some(&args[0]),
                    end: Some(&args[1]),
                    limits: ast::RangeLimits::Closed,
                })
            },
            hir::ExprKind::Struct(path, fields, None) => match &path {
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
                            return Some(VecArgs::Vec(args));
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

/// A desugared `while` loop
pub struct While<'hir> {
    /// `while` loop condition
    pub condition: &'hir Expr<'hir>,
    /// `while` loop body
    pub body: &'hir Expr<'hir>,
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
            _,
        ) = expr.kind
        {
            return Some(Self { condition, body });
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
                                    kind: ExprKind::Let(let_pat, let_expr, _),
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
            if let ExprKind::Call(_, args) = expr.kind;
            if let Some((strs_ref, args, fmt_expr)) = match args {
                // Arguments::new_v1
                [strs_ref, args] => Some((strs_ref, args, None)),
                // Arguments::new_v1_formatted
                [strs_ref, args, fmt_expr, _unsafe_arg] => Some((strs_ref, args, Some(fmt_expr))),
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

    /// Returns a vector of `FormatArgsArg`.
    pub fn args(&self) -> Option<Vec<FormatArgsArg<'tcx>>> {
        if let Some(expr) = self.fmt_expr {
            if_chain! {
                if let ExprKind::AddrOf(BorrowKind::Ref, _, expr) = expr.kind;
                if let ExprKind::Array(exprs) = expr.kind;
                then {
                    exprs.iter().map(|fmt| {
                        if_chain! {
                            // struct `core::fmt::rt::v1::Argument`
                            if let ExprKind::Struct(_, fields, _) = fmt.kind;
                            if let Some(position_field) = fields.iter().find(|f| f.ident.name == sym::position);
                            if let ExprKind::Lit(lit) = &position_field.expr.kind;
                            if let LitKind::Int(position, _) = lit.node;
                            if let Ok(i) = usize::try_from(position);
                            let arg = &self.args[i];
                            if let ExprKind::Call(_, [arg_name, _]) = arg.kind;
                            if let ExprKind::Field(_, j) = arg_name.kind;
                            if let Ok(j) = j.name.as_str().parse::<usize>();
                            then {
                                Some(FormatArgsArg { value: self.value_args[j], arg, fmt: Some(fmt) })
                            } else {
                                None
                            }
                        }
                    }).collect()
                } else {
                    None
                }
            }
        } else {
            Some(
                self.value_args
                    .iter()
                    .zip(self.args.iter())
                    .map(|(value, arg)| FormatArgsArg { value, arg, fmt: None })
                    .collect(),
            )
        }
    }
}

/// Type representing a `FormatArgsExpn`'s format arguments
pub struct FormatArgsArg<'tcx> {
    /// An element of `value_args` according to `position`
    pub value: &'tcx Expr<'tcx>,
    /// An element of `args` according to `position`
    pub arg: &'tcx Expr<'tcx>,
    /// An element of `fmt_expn`
    pub fmt: Option<&'tcx Expr<'tcx>>,
}

impl<'tcx> FormatArgsArg<'tcx> {
    /// Returns true if any formatting parameters are used that would have an effect on strings,
    /// like `{:+2}` instead of just `{}`.
    pub fn has_string_formatting(&self) -> bool {
        self.fmt.map_or(false, |fmt| {
            // `!` because these conditions check that `self` is unformatted.
            !if_chain! {
                // struct `core::fmt::rt::v1::Argument`
                if let ExprKind::Struct(_, fields, _) = fmt.kind;
                if let Some(format_field) = fields.iter().find(|f| f.ident.name == sym::format);
                // struct `core::fmt::rt::v1::FormatSpec`
                if let ExprKind::Struct(_, subfields, _) = format_field.expr.kind;
                let mut precision_found = false;
                let mut width_found = false;
                if subfields.iter().all(|field| {
                    match field.ident.name {
                        sym::precision => {
                            precision_found = true;
                            if let ExprKind::Path(ref precision_path) = field.expr.kind {
                                last_path_segment(precision_path).ident.name == sym::Implied
                            } else {
                                false
                            }
                        }
                        sym::width => {
                            width_found = true;
                            if let ExprKind::Path(ref width_qpath) = field.expr.kind {
                                last_path_segment(width_qpath).ident.name == sym::Implied
                            } else {
                                false
                            }
                        }
                        _ => true,
                    }
                });
                if precision_found && width_found;
                then { true } else { false }
            }
        })
    }

    /// Returns true if the argument is formatted using `Display::fmt`.
    pub fn is_display(&self) -> bool {
        if_chain! {
            if let ExprKind::Call(_, [_, format_field]) = self.arg.kind;
            if let ExprKind::Path(QPath::Resolved(_, path)) = format_field.kind;
            if let [.., t, _] = path.segments;
            if t.ident.name == sym::Display;
            then { true } else { false }
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

/// A parsed `panic!` expansion
pub struct PanicExpn<'tcx> {
    /// Span of `panic!(..)`
    pub call_site: Span,
    /// Inner `format_args!` expansion
    pub format_args: FormatArgsExpn<'tcx>,
}

impl PanicExpn<'tcx> {
    /// Parses an expanded `panic!` invocation
    pub fn parse(expr: &'tcx Expr<'tcx>) -> Option<Self> {
        if_chain! {
            if let ExprKind::Call(_, [format_args]) = expr.kind;
            let expn_data = expr.span.ctxt().outer_expn_data();
            if let Some(format_args) = FormatArgsExpn::parse(format_args);
            then {
                Some(PanicExpn {
                    call_site: expn_data.call_site,
                    format_args,
                })
            } else {
                None
            }
        }
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
    WithLiteralCapacity(u64),
    /// `Vec::with_capacity(slice.len())`
    WithExprCapacity(HirId),
}

/// Checks if given expression is an initialization of `Vec` and returns its kind.
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
                    if_chain! {
                        if let ExprKind::Lit(lit) = &arg.kind;
                        if let LitKind::Int(num, _) = lit.node;
                        then {
                            return Some(VecInitKind::WithLiteralCapacity(num.try_into().ok()?))
                        }
                    }
                    return Some(VecInitKind::WithExprCapacity(arg.hir_id));
                }
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
