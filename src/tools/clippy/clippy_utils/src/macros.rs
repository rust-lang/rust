#![allow(clippy::similar_names)] // `expr` and `expn`

use crate::is_path_diagnostic_item;
use crate::source::snippet_opt;
use crate::visitors::expr_visitor_no_bodies;

use arrayvec::ArrayVec;
use itertools::{izip, Either, Itertools};
use rustc_ast::ast::LitKind;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, Expr, ExprKind, HirId, Node, QPath};
use rustc_lexer::unescape::unescape_literal;
use rustc_lexer::{tokenize, unescape, LiteralKind, TokenKind};
use rustc_lint::LateContext;
use rustc_parse_format::{self as rpf, Alignment};
use rustc_span::def_id::DefId;
use rustc_span::hygiene::{self, MacroKind, SyntaxContext};
use rustc_span::{sym, BytePos, ExpnData, ExpnId, ExpnKind, Pos, Span, SpanData, Symbol};
use std::ops::ControlFlow;

const FORMAT_MACRO_DIAG_ITEMS: &[Symbol] = &[
    sym::assert_eq_macro,
    sym::assert_macro,
    sym::assert_ne_macro,
    sym::debug_assert_eq_macro,
    sym::debug_assert_macro,
    sym::debug_assert_ne_macro,
    sym::eprint_macro,
    sym::eprintln_macro,
    sym::format_args_macro,
    sym::format_macro,
    sym::print_macro,
    sym::println_macro,
    sym::std_panic_macro,
    sym::write_macro,
    sym::writeln_macro,
];

/// Returns true if a given Macro `DefId` is a format macro (e.g. `println!`)
pub fn is_format_macro(cx: &LateContext<'_>, macro_def_id: DefId) -> bool {
    if let Some(name) = cx.tcx.get_diagnostic_name(macro_def_id) {
        FORMAT_MACRO_DIAG_ITEMS.contains(&name)
    } else {
        false
    }
}

/// A macro call, like `vec![1, 2, 3]`.
///
/// Use `tcx.item_name(macro_call.def_id)` to get the macro name.
/// Even better is to check if it is a diagnostic item.
///
/// This structure is similar to `ExpnData` but it precludes desugaring expansions.
#[derive(Debug)]
pub struct MacroCall {
    /// Macro `DefId`
    pub def_id: DefId,
    /// Kind of macro
    pub kind: MacroKind,
    /// The expansion produced by the macro call
    pub expn: ExpnId,
    /// Span of the macro call site
    pub span: Span,
}

impl MacroCall {
    pub fn is_local(&self) -> bool {
        span_is_local(self.span)
    }
}

/// Returns an iterator of expansions that created the given span
pub fn expn_backtrace(mut span: Span) -> impl Iterator<Item = (ExpnId, ExpnData)> {
    std::iter::from_fn(move || {
        let ctxt = span.ctxt();
        if ctxt == SyntaxContext::root() {
            return None;
        }
        let expn = ctxt.outer_expn();
        let data = expn.expn_data();
        span = data.call_site;
        Some((expn, data))
    })
}

/// Checks whether the span is from the root expansion or a locally defined macro
pub fn span_is_local(span: Span) -> bool {
    !span.from_expansion() || expn_is_local(span.ctxt().outer_expn())
}

/// Checks whether the expansion is the root expansion or a locally defined macro
pub fn expn_is_local(expn: ExpnId) -> bool {
    if expn == ExpnId::root() {
        return true;
    }
    let data = expn.expn_data();
    let backtrace = expn_backtrace(data.call_site);
    std::iter::once((expn, data))
        .chain(backtrace)
        .find_map(|(_, data)| data.macro_def_id)
        .map_or(true, DefId::is_local)
}

/// Returns an iterator of macro expansions that created the given span.
/// Note that desugaring expansions are skipped.
pub fn macro_backtrace(span: Span) -> impl Iterator<Item = MacroCall> {
    expn_backtrace(span).filter_map(|(expn, data)| match data {
        ExpnData {
            kind: ExpnKind::Macro(kind, _),
            macro_def_id: Some(def_id),
            call_site: span,
            ..
        } => Some(MacroCall {
            def_id,
            kind,
            expn,
            span,
        }),
        _ => None,
    })
}

/// If the macro backtrace of `span` has a macro call at the root expansion
/// (i.e. not a nested macro call), returns `Some` with the `MacroCall`
pub fn root_macro_call(span: Span) -> Option<MacroCall> {
    macro_backtrace(span).last()
}

/// Like [`root_macro_call`], but only returns `Some` if `node` is the "first node"
/// produced by the macro call, as in [`first_node_in_macro`].
pub fn root_macro_call_first_node(cx: &LateContext<'_>, node: &impl HirNode) -> Option<MacroCall> {
    if first_node_in_macro(cx, node) != Some(ExpnId::root()) {
        return None;
    }
    root_macro_call(node.span())
}

/// Like [`macro_backtrace`], but only returns macro calls where `node` is the "first node" of the
/// macro call, as in [`first_node_in_macro`].
pub fn first_node_macro_backtrace(cx: &LateContext<'_>, node: &impl HirNode) -> impl Iterator<Item = MacroCall> {
    let span = node.span();
    first_node_in_macro(cx, node)
        .into_iter()
        .flat_map(move |expn| macro_backtrace(span).take_while(move |macro_call| macro_call.expn != expn))
}

/// If `node` is the "first node" in a macro expansion, returns `Some` with the `ExpnId` of the
/// macro call site (i.e. the parent of the macro expansion). This generally means that `node`
/// is the outermost node of an entire macro expansion, but there are some caveats noted below.
/// This is useful for finding macro calls while visiting the HIR without processing the macro call
/// at every node within its expansion.
///
/// If you already have immediate access to the parent node, it is simpler to
/// just check the context of that span directly (e.g. `parent.span.from_expansion()`).
///
/// If a macro call is in statement position, it expands to one or more statements.
/// In that case, each statement *and* their immediate descendants will all yield `Some`
/// with the `ExpnId` of the containing block.
///
/// A node may be the "first node" of multiple macro calls in a macro backtrace.
/// The expansion of the outermost macro call site is returned in such cases.
pub fn first_node_in_macro(cx: &LateContext<'_>, node: &impl HirNode) -> Option<ExpnId> {
    // get the macro expansion or return `None` if not found
    // `macro_backtrace` importantly ignores desugaring expansions
    let expn = macro_backtrace(node.span()).next()?.expn;

    // get the parent node, possibly skipping over a statement
    // if the parent is not found, it is sensible to return `Some(root)`
    let hir = cx.tcx.hir();
    let mut parent_iter = hir.parent_iter(node.hir_id());
    let (parent_id, _) = match parent_iter.next() {
        None => return Some(ExpnId::root()),
        Some((_, Node::Stmt(_))) => match parent_iter.next() {
            None => return Some(ExpnId::root()),
            Some(next) => next,
        },
        Some(next) => next,
    };

    // get the macro expansion of the parent node
    let parent_span = hir.span(parent_id);
    let Some(parent_macro_call) = macro_backtrace(parent_span).next() else {
        // the parent node is not in a macro
        return Some(ExpnId::root());
    };

    if parent_macro_call.expn.is_descendant_of(expn) {
        // `node` is input to a macro call
        return None;
    }

    Some(parent_macro_call.expn)
}

/* Specific Macro Utils */

/// Is `def_id` of `std::panic`, `core::panic` or any inner implementation macros
pub fn is_panic(cx: &LateContext<'_>, def_id: DefId) -> bool {
    let Some(name) = cx.tcx.get_diagnostic_name(def_id) else { return false };
    matches!(
        name.as_str(),
        "core_panic_macro"
            | "std_panic_macro"
            | "core_panic_2015_macro"
            | "std_panic_2015_macro"
            | "core_panic_2021_macro"
    )
}

pub enum PanicExpn<'a> {
    /// No arguments - `panic!()`
    Empty,
    /// A string literal or any `&str` - `panic!("message")` or `panic!(message)`
    Str(&'a Expr<'a>),
    /// A single argument that implements `Display` - `panic!("{}", object)`
    Display(&'a Expr<'a>),
    /// Anything else - `panic!("error {}: {}", a, b)`
    Format(FormatArgsExpn<'a>),
}

impl<'a> PanicExpn<'a> {
    pub fn parse(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<Self> {
        if !macro_backtrace(expr.span).any(|macro_call| is_panic(cx, macro_call.def_id)) {
            return None;
        }
        let ExprKind::Call(callee, [arg]) = &expr.kind else { return None };
        let ExprKind::Path(QPath::Resolved(_, path)) = &callee.kind else { return None };
        let result = match path.segments.last().unwrap().ident.as_str() {
            "panic" if arg.span.ctxt() == expr.span.ctxt() => Self::Empty,
            "panic" | "panic_str" => Self::Str(arg),
            "panic_display" => {
                let ExprKind::AddrOf(_, _, e) = &arg.kind else { return None };
                Self::Display(e)
            },
            "panic_fmt" => Self::Format(FormatArgsExpn::parse(cx, arg)?),
            _ => return None,
        };
        Some(result)
    }
}

/// Finds the arguments of an `assert!` or `debug_assert!` macro call within the macro expansion
pub fn find_assert_args<'a>(
    cx: &LateContext<'_>,
    expr: &'a Expr<'a>,
    expn: ExpnId,
) -> Option<(&'a Expr<'a>, PanicExpn<'a>)> {
    find_assert_args_inner(cx, expr, expn).map(|([e], p)| (e, p))
}

/// Finds the arguments of an `assert_eq!` or `debug_assert_eq!` macro call within the macro
/// expansion
pub fn find_assert_eq_args<'a>(
    cx: &LateContext<'_>,
    expr: &'a Expr<'a>,
    expn: ExpnId,
) -> Option<(&'a Expr<'a>, &'a Expr<'a>, PanicExpn<'a>)> {
    find_assert_args_inner(cx, expr, expn).map(|([a, b], p)| (a, b, p))
}

fn find_assert_args_inner<'a, const N: usize>(
    cx: &LateContext<'_>,
    expr: &'a Expr<'a>,
    expn: ExpnId,
) -> Option<([&'a Expr<'a>; N], PanicExpn<'a>)> {
    let macro_id = expn.expn_data().macro_def_id?;
    let (expr, expn) = match cx.tcx.item_name(macro_id).as_str().strip_prefix("debug_") {
        None => (expr, expn),
        Some(inner_name) => find_assert_within_debug_assert(cx, expr, expn, Symbol::intern(inner_name))?,
    };
    let mut args = ArrayVec::new();
    let mut panic_expn = None;
    expr_visitor_no_bodies(|e| {
        if args.is_full() {
            if panic_expn.is_none() && e.span.ctxt() != expr.span.ctxt() {
                panic_expn = PanicExpn::parse(cx, e);
            }
            panic_expn.is_none()
        } else if is_assert_arg(cx, e, expn) {
            args.push(e);
            false
        } else {
            true
        }
    })
    .visit_expr(expr);
    let args = args.into_inner().ok()?;
    // if no `panic!(..)` is found, use `PanicExpn::Empty`
    // to indicate that the default assertion message is used
    let panic_expn = panic_expn.unwrap_or(PanicExpn::Empty);
    Some((args, panic_expn))
}

fn find_assert_within_debug_assert<'a>(
    cx: &LateContext<'_>,
    expr: &'a Expr<'a>,
    expn: ExpnId,
    assert_name: Symbol,
) -> Option<(&'a Expr<'a>, ExpnId)> {
    let mut found = None;
    expr_visitor_no_bodies(|e| {
        if found.is_some() || !e.span.from_expansion() {
            return false;
        }
        let e_expn = e.span.ctxt().outer_expn();
        if e_expn == expn {
            return true;
        }
        if e_expn.expn_data().macro_def_id.map(|id| cx.tcx.item_name(id)) == Some(assert_name) {
            found = Some((e, e_expn));
        }
        false
    })
    .visit_expr(expr);
    found
}

fn is_assert_arg(cx: &LateContext<'_>, expr: &Expr<'_>, assert_expn: ExpnId) -> bool {
    if !expr.span.from_expansion() {
        return true;
    }
    let result = macro_backtrace(expr.span).try_for_each(|macro_call| {
        if macro_call.expn == assert_expn {
            ControlFlow::Break(false)
        } else {
            match cx.tcx.item_name(macro_call.def_id) {
                // `cfg!(debug_assertions)` in `debug_assert!`
                sym::cfg => ControlFlow::CONTINUE,
                // assert!(other_macro!(..))
                _ => ControlFlow::Break(true),
            }
        }
    });
    match result {
        ControlFlow::Break(is_assert_arg) => is_assert_arg,
        ControlFlow::Continue(()) => true,
    }
}

/// The format string doesn't exist in the HIR, so we reassemble it from source code
#[derive(Debug)]
pub struct FormatString {
    /// Span of the whole format string literal, including `[r#]"`.
    pub span: Span,
    /// Snippet of the whole format string literal, including `[r#]"`.
    pub snippet: String,
    /// If the string is raw `r"..."`/`r#""#`, how many `#`s does it have on each side.
    pub style: Option<usize>,
    /// The unescaped value of the format string, e.g. `"val – {}"` for the literal
    /// `"val \u{2013} {}"`.
    pub unescaped: String,
    /// The format string split by format args like `{..}`.
    pub parts: Vec<Symbol>,
}

impl FormatString {
    fn new(cx: &LateContext<'_>, pieces: &Expr<'_>) -> Option<Self> {
        // format_args!(r"a {} b \", 1);
        //
        // expands to
        //
        // ::core::fmt::Arguments::new_v1(&["a ", " b \\"],
        //      &[::core::fmt::ArgumentV1::new_display(&1)]);
        //
        // where `pieces` is the expression `&["a ", " b \\"]`. It has the span of `r"a {} b \"`
        let span = pieces.span;
        let snippet = snippet_opt(cx, span)?;

        let (inner, style) = match tokenize(&snippet).next()?.kind {
            TokenKind::Literal { kind, .. } => {
                let style = match kind {
                    LiteralKind::Str { .. } => None,
                    LiteralKind::RawStr { n_hashes: Some(n), .. } => Some(n.into()),
                    _ => return None,
                };

                let start = style.map_or(1, |n| 2 + n);
                let end = snippet.len() - style.map_or(1, |n| 1 + n);

                (&snippet[start..end], style)
            },
            _ => return None,
        };

        let mode = if style.is_some() {
            unescape::Mode::RawStr
        } else {
            unescape::Mode::Str
        };

        let mut unescaped = String::with_capacity(inner.len());
        unescape_literal(inner, mode, &mut |_, ch| match ch {
            Ok(ch) => unescaped.push(ch),
            Err(e) if !e.is_fatal() => (),
            Err(e) => panic!("{:?}", e),
        });

        let mut parts = Vec::new();
        expr_visitor_no_bodies(|expr| {
            if let ExprKind::Lit(lit) = &expr.kind {
                if let LitKind::Str(symbol, _) = lit.node {
                    parts.push(symbol);
                }
            }

            true
        })
        .visit_expr(pieces);

        Some(Self {
            span,
            snippet,
            style,
            unescaped,
            parts,
        })
    }
}

struct FormatArgsValues<'tcx> {
    /// See `FormatArgsExpn::value_args`
    value_args: Vec<&'tcx Expr<'tcx>>,
    /// Maps an `rt::v1::Argument::position` or an `rt::v1::Count::Param` to its index in
    /// `value_args`
    pos_to_value_index: Vec<usize>,
    /// Used to check if a value is declared inline & to resolve `InnerSpan`s.
    format_string_span: SpanData,
}

impl<'tcx> FormatArgsValues<'tcx> {
    fn new(args: &'tcx Expr<'tcx>, format_string_span: SpanData) -> Self {
        let mut pos_to_value_index = Vec::new();
        let mut value_args = Vec::new();
        expr_visitor_no_bodies(|expr| {
            if expr.span.ctxt() == args.span.ctxt() {
                // ArgumentV1::new_<format_trait>(<val>)
                // ArgumentV1::from_usize(<val>)
                if let ExprKind::Call(callee, [val]) = expr.kind
                    && let ExprKind::Path(QPath::TypeRelative(ty, _)) = callee.kind
                    && let hir::TyKind::Path(QPath::Resolved(_, path)) = ty.kind
                    && path.segments.last().unwrap().ident.name == sym::ArgumentV1
                {
                    let val_idx = if val.span.ctxt() == expr.span.ctxt()
                        && let ExprKind::Field(_, field) = val.kind
                        && let Ok(idx) = field.name.as_str().parse()
                    {
                        // tuple index
                        idx
                    } else {
                        // assume the value expression is passed directly
                        pos_to_value_index.len()
                    };

                    pos_to_value_index.push(val_idx);
                }

                true
            } else {
                // assume that any expr with a differing span is a value
                value_args.push(expr);

                false
            }
        })
        .visit_expr(args);

        Self {
            value_args,
            pos_to_value_index,
            format_string_span,
        }
    }
}

/// The positions of a format argument's value, precision and width
///
/// A position is an index into the second argument of `Arguments::new_v1[_formatted]`
#[derive(Debug, Default, Copy, Clone)]
struct ParamPosition {
    /// The position stored in `rt::v1::Argument::position`.
    value: usize,
    /// The position stored in `rt::v1::FormatSpec::width` if it is a `Count::Param`.
    width: Option<usize>,
    /// The position stored in `rt::v1::FormatSpec::precision` if it is a `Count::Param`.
    precision: Option<usize>,
}

/// Parses the `fmt` arg of `Arguments::new_v1_formatted(pieces, args, fmt, _)`
fn parse_rt_fmt<'tcx>(fmt_arg: &'tcx Expr<'tcx>) -> Option<impl Iterator<Item = ParamPosition> + 'tcx> {
    fn parse_count(expr: &Expr<'_>) -> Option<usize> {
        // ::core::fmt::rt::v1::Count::Param(1usize),
        if let ExprKind::Call(ctor, [val]) = expr.kind
            && let ExprKind::Path(QPath::Resolved(_, path)) = ctor.kind
            && path.segments.last()?.ident.name == sym::Param
            && let ExprKind::Lit(lit) = &val.kind
            && let LitKind::Int(pos, _) = lit.node
        {
            Some(pos as usize)
        } else {
            None
        }
    }

    if let ExprKind::AddrOf(.., array) = fmt_arg.kind
        && let ExprKind::Array(specs) = array.kind
    {
        Some(specs.iter().map(|spec| {
            let mut position = ParamPosition::default();

            // ::core::fmt::rt::v1::Argument {
            //     position: 0usize,
            //     format: ::core::fmt::rt::v1::FormatSpec {
            //         ..
            //         precision: ::core::fmt::rt::v1::Count::Implied,
            //         width: ::core::fmt::rt::v1::Count::Implied,
            //     },
            // }

            // TODO: this can be made much nicer next sync with `Visitor::visit_expr_field`
            if let ExprKind::Struct(_, fields, _) = spec.kind {
                for field in fields {
                    match (field.ident.name, &field.expr.kind) {
                        (sym::position, ExprKind::Lit(lit)) => {
                            if let LitKind::Int(pos, _) = lit.node {
                                position.value = pos as usize;
                            }
                        },
                        (sym::format, &ExprKind::Struct(_, spec_fields, _)) => {
                            for spec_field in spec_fields {
                                match spec_field.ident.name {
                                    sym::precision => {
                                        position.precision = parse_count(spec_field.expr);
                                    },
                                    sym::width => {
                                        position.width = parse_count(spec_field.expr);
                                    },
                                    _ => {},
                                }
                            }
                        },
                        _ => {},
                    }
                }
            }

            position
        }))
    } else {
        None
    }
}

/// `Span::from_inner`, but for `rustc_parse_format`'s `InnerSpan`
fn span_from_inner(base: SpanData, inner: rpf::InnerSpan) -> Span {
    Span::new(
        base.lo + BytePos::from_usize(inner.start),
        base.lo + BytePos::from_usize(inner.end),
        base.ctxt,
        base.parent,
    )
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FormatParamKind {
    /// An implicit parameter , such as `{}` or `{:?}`.
    Implicit,
    /// A parameter with an explicit number, or an asterisk precision. e.g. `{1}`, `{0:?}`,
    /// `{:.0$}` or `{:.*}`.
    Numbered,
    /// A named parameter with a named `value_arg`, such as the `x` in `format!("{x}", x = 1)`.
    Named(Symbol),
    /// An implicit named parameter, such as the `y` in `format!("{y}")`.
    NamedInline(Symbol),
}

/// A `FormatParam` is any place in a `FormatArgument` that refers to a supplied value, e.g.
///
/// ```
/// let precision = 2;
/// format!("{:.precision$}", 0.1234);
/// ```
///
/// has two `FormatParam`s, a [`FormatParamKind::Implicit`] `.kind` with a `.value` of `0.1234`
/// and a [`FormatParamKind::NamedInline("precision")`] `.kind` with a `.value` of `2`
#[derive(Debug, Copy, Clone)]
pub struct FormatParam<'tcx> {
    /// The expression this parameter refers to.
    pub value: &'tcx Expr<'tcx>,
    /// How this parameter refers to its `value`.
    pub kind: FormatParamKind,
    /// Span of the parameter, may be zero width. Includes the whitespace of implicit parameters.
    ///
    /// ```text
    /// format!("{}, {  }, {0}, {name}", ...);
    ///          ^    ~~    ~    ~~~~
    /// ```
    pub span: Span,
}

impl<'tcx> FormatParam<'tcx> {
    fn new(
        mut kind: FormatParamKind,
        position: usize,
        inner: rpf::InnerSpan,
        values: &FormatArgsValues<'tcx>,
    ) -> Option<Self> {
        let value_index = *values.pos_to_value_index.get(position)?;
        let value = *values.value_args.get(value_index)?;
        let span = span_from_inner(values.format_string_span, inner);

        // if a param is declared inline, e.g. `format!("{x}")`, the generated expr's span points
        // into the format string
        if let FormatParamKind::Named(name) = kind && values.format_string_span.contains(value.span.data()) {
            kind = FormatParamKind::NamedInline(name);
        }

        Some(Self { value, kind, span })
    }
}

/// Used by [width](https://doc.rust-lang.org/std/fmt/#width) and
/// [precision](https://doc.rust-lang.org/std/fmt/#precision) specifiers.
#[derive(Debug, Copy, Clone)]
pub enum Count<'tcx> {
    /// Specified with a literal number, stores the value.
    Is(usize, Span),
    /// Specified using `$` and `*` syntaxes. The `*` format is still considered to be
    /// `FormatParamKind::Numbered`.
    Param(FormatParam<'tcx>),
    /// Not specified.
    Implied,
}

impl<'tcx> Count<'tcx> {
    fn new(
        count: rpf::Count<'_>,
        position: Option<usize>,
        inner: Option<rpf::InnerSpan>,
        values: &FormatArgsValues<'tcx>,
    ) -> Option<Self> {
        Some(match count {
            rpf::Count::CountIs(val) => Self::Is(val, span_from_inner(values.format_string_span, inner?)),
            rpf::Count::CountIsName(name, span) => Self::Param(FormatParam::new(
                FormatParamKind::Named(Symbol::intern(name)),
                position?,
                span,
                values,
            )?),
            rpf::Count::CountIsParam(_) | rpf::Count::CountIsStar(_) => {
                Self::Param(FormatParam::new(FormatParamKind::Numbered, position?, inner?, values)?)
            },
            rpf::Count::CountImplied => Self::Implied,
        })
    }

    pub fn is_implied(self) -> bool {
        matches!(self, Count::Implied)
    }

    pub fn param(self) -> Option<FormatParam<'tcx>> {
        match self {
            Count::Param(param) => Some(param),
            _ => None,
        }
    }
}

/// Specification for the formatting of an argument in the format string. See
/// <https://doc.rust-lang.org/std/fmt/index.html#formatting-parameters> for the precise meanings.
#[derive(Debug)]
pub struct FormatSpec<'tcx> {
    /// Optionally specified character to fill alignment with.
    pub fill: Option<char>,
    /// Optionally specified alignment.
    pub align: Alignment,
    /// Packed version of various flags provided, see [`rustc_parse_format::Flag`].
    pub flags: u32,
    /// Represents either the maximum width or the integer precision.
    pub precision: Count<'tcx>,
    /// The minimum width, will be padded according to `width`/`align`
    pub width: Count<'tcx>,
    /// The formatting trait used by the argument, e.g. `sym::Display` for `{}`, `sym::Debug` for
    /// `{:?}`.
    pub r#trait: Symbol,
    pub trait_span: Option<Span>,
}

impl<'tcx> FormatSpec<'tcx> {
    fn new(spec: rpf::FormatSpec<'_>, positions: ParamPosition, values: &FormatArgsValues<'tcx>) -> Option<Self> {
        Some(Self {
            fill: spec.fill,
            align: spec.align,
            flags: spec.flags,
            precision: Count::new(spec.precision, positions.precision, spec.precision_span, values)?,
            width: Count::new(spec.width, positions.width, spec.width_span, values)?,
            r#trait: match spec.ty {
                "" => sym::Display,
                "?" => sym::Debug,
                "o" => sym!(Octal),
                "x" => sym!(LowerHex),
                "X" => sym!(UpperHex),
                "p" => sym::Pointer,
                "b" => sym!(Binary),
                "e" => sym!(LowerExp),
                "E" => sym!(UpperExp),
                _ => return None,
            },
            trait_span: spec
                .ty_span
                .map(|span| span_from_inner(values.format_string_span, span)),
        })
    }

    /// Returns true if this format spec would change the contents of a string when formatted
    pub fn has_string_formatting(&self) -> bool {
        self.r#trait != sym::Display || !self.width.is_implied() || !self.precision.is_implied()
    }
}

/// A format argument, such as `{}`, `{foo:?}`.
#[derive(Debug)]
pub struct FormatArg<'tcx> {
    /// The parameter the argument refers to.
    pub param: FormatParam<'tcx>,
    /// How to format `param`.
    pub format: FormatSpec<'tcx>,
    /// span of the whole argument, `{..}`.
    pub span: Span,
}

/// A parsed `format_args!` expansion.
#[derive(Debug)]
pub struct FormatArgsExpn<'tcx> {
    /// The format string literal.
    pub format_string: FormatString,
    // The format arguments, such as `{:?}`.
    pub args: Vec<FormatArg<'tcx>>,
    /// Has an added newline due to `println!()`/`writeln!()`/etc. The last format string part will
    /// include this added newline.
    pub newline: bool,
    /// Values passed after the format string and implicit captures. `[1, z + 2, x]` for
    /// `format!("{x} {} {y}", 1, z + 2)`.
    value_args: Vec<&'tcx Expr<'tcx>>,
}

impl<'tcx> FormatArgsExpn<'tcx> {
    pub fn parse(cx: &LateContext<'_>, expr: &'tcx Expr<'tcx>) -> Option<Self> {
        let macro_name = macro_backtrace(expr.span)
            .map(|macro_call| cx.tcx.item_name(macro_call.def_id))
            .find(|&name| matches!(name, sym::const_format_args | sym::format_args | sym::format_args_nl))?;
        let newline = macro_name == sym::format_args_nl;

        // ::core::fmt::Arguments::new_v1(pieces, args)
        // ::core::fmt::Arguments::new_v1_formatted(pieces, args, fmt, _unsafe_arg)
        if let ExprKind::Call(callee, [pieces, args, rest @ ..]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = callee.kind
            && is_path_diagnostic_item(cx, ty, sym::Arguments)
            && matches!(seg.ident.as_str(), "new_v1" | "new_v1_formatted")
        {
            let format_string = FormatString::new(cx, pieces)?;

            let mut parser = rpf::Parser::new(
                &format_string.unescaped,
                format_string.style,
                Some(format_string.snippet.clone()),
                // `format_string.unescaped` does not contain the appended newline
                false,
                rpf::ParseMode::Format,
            );

            let parsed_args = parser
                .by_ref()
                .filter_map(|piece| match piece {
                    rpf::Piece::NextArgument(a) => Some(a),
                    rpf::Piece::String(_) => None,
                })
                .collect_vec();
            if !parser.errors.is_empty() {
                return None;
            }

            let positions = if let Some(fmt_arg) = rest.first() {
                // If the argument contains format specs, `new_v1_formatted(_, _, fmt, _)`, parse
                // them.

                Either::Left(parse_rt_fmt(fmt_arg)?)
            } else {
                // If no format specs are given, the positions are in the given order and there are
                // no `precision`/`width`s to consider.

                Either::Right((0..).map(|n| ParamPosition {
                    value: n,
                    width: None,
                    precision: None,
                }))
            };

            let values = FormatArgsValues::new(args, format_string.span.data());

            let args = izip!(positions, parsed_args, parser.arg_places)
                .map(|(position, parsed_arg, arg_span)| {
                    Some(FormatArg {
                        param: FormatParam::new(
                            match parsed_arg.position {
                                rpf::Position::ArgumentImplicitlyIs(_) => FormatParamKind::Implicit,
                                rpf::Position::ArgumentIs(_) => FormatParamKind::Numbered,
                                // NamedInline is handled by `FormatParam::new()`
                                rpf::Position::ArgumentNamed(name) => FormatParamKind::Named(Symbol::intern(name)),
                            },
                            position.value,
                            parsed_arg.position_span,
                            &values,
                        )?,
                        format: FormatSpec::new(parsed_arg.format, position, &values)?,
                        span: span_from_inner(values.format_string_span, arg_span),
                    })
                })
                .collect::<Option<Vec<_>>>()?;

            Some(Self {
                format_string,
                args,
                value_args: values.value_args,
                newline,
            })
        } else {
            None
        }
    }

    pub fn find_nested(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, expn_id: ExpnId) -> Option<Self> {
        let mut format_args = None;
        expr_visitor_no_bodies(|e| {
            if format_args.is_some() {
                return false;
            }
            let e_ctxt = e.span.ctxt();
            if e_ctxt == expr.span.ctxt() {
                return true;
            }
            if e_ctxt.outer_expn().is_descendant_of(expn_id) {
                format_args = FormatArgsExpn::parse(cx, e);
            }
            false
        })
        .visit_expr(expr);
        format_args
    }

    /// Source callsite span of all inputs
    pub fn inputs_span(&self) -> Span {
        match *self.value_args {
            [] => self.format_string.span,
            [.., last] => self
                .format_string
                .span
                .to(hygiene::walk_chain(last.span, self.format_string.span.ctxt())),
        }
    }

    /// Iterator of all format params, both values and those referenced by `width`/`precision`s.
    pub fn params(&'tcx self) -> impl Iterator<Item = FormatParam<'tcx>> {
        self.args
            .iter()
            .flat_map(|arg| [Some(arg.param), arg.format.precision.param(), arg.format.width.param()])
            .flatten()
    }
}

/// A node with a `HirId` and a `Span`
pub trait HirNode {
    fn hir_id(&self) -> HirId;
    fn span(&self) -> Span;
}

macro_rules! impl_hir_node {
    ($($t:ident),*) => {
        $(impl HirNode for hir::$t<'_> {
            fn hir_id(&self) -> HirId {
                self.hir_id
            }
            fn span(&self) -> Span {
                self.span
            }
        })*
    };
}

impl_hir_node!(Expr, Pat);

impl HirNode for hir::Item<'_> {
    fn hir_id(&self) -> HirId {
        self.hir_id()
    }

    fn span(&self) -> Span {
        self.span
    }
}
