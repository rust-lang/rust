#![allow(clippy::similar_names)] // `expr` and `expn`

use crate::source::snippet_opt;
use crate::visitors::{for_each_expr, Descend};

use arrayvec::ArrayVec;
use itertools::{izip, Either, Itertools};
use rustc_ast::ast::LitKind;
use rustc_ast::FormatArgs;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{self as hir, Expr, ExprField, ExprKind, HirId, LangItem, Node, QPath, TyKind};
use rustc_lexer::unescape::unescape_literal;
use rustc_lexer::{tokenize, unescape, LiteralKind, TokenKind};
use rustc_lint::LateContext;
use rustc_parse_format::{self as rpf, Alignment};
use rustc_span::def_id::DefId;
use rustc_span::hygiene::{self, MacroKind, SyntaxContext};
use rustc_span::{sym, BytePos, ExpnData, ExpnId, ExpnKind, Pos, Span, SpanData, Symbol};
use std::cell::RefCell;
use std::iter::{once, zip};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};

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
        name,
        sym::core_panic_macro
            | sym::std_panic_macro
            | sym::core_panic_2015_macro
            | sym::std_panic_2015_macro
            | sym::core_panic_2021_macro
    )
}

/// Is `def_id` of `assert!` or `debug_assert!`
pub fn is_assert_macro(cx: &LateContext<'_>, def_id: DefId) -> bool {
    let Some(name) = cx.tcx.get_diagnostic_name(def_id) else { return false };
    matches!(name, sym::assert_macro | sym::debug_assert_macro)
}

#[derive(Debug)]
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
        let ExprKind::Call(callee, [arg, rest @ ..]) = &expr.kind else { return None };
        let ExprKind::Path(QPath::Resolved(_, path)) = &callee.kind else { return None };
        let result = match path.segments.last().unwrap().ident.as_str() {
            "panic" if arg.span.ctxt() == expr.span.ctxt() => Self::Empty,
            "panic" | "panic_str" => Self::Str(arg),
            "panic_display" => {
                let ExprKind::AddrOf(_, _, e) = &arg.kind else { return None };
                Self::Display(e)
            },
            "panic_fmt" => Self::Format(FormatArgsExpn::parse(cx, arg)?),
            // Since Rust 1.52, `assert_{eq,ne}` macros expand to use:
            // `core::panicking::assert_failed(.., left_val, right_val, None | Some(format_args!(..)));`
            "assert_failed" => {
                // It should have 4 arguments in total (we already matched with the first argument,
                // so we're just checking for 3)
                if rest.len() != 3 {
                    return None;
                }
                // `msg_arg` is either `None` (no custom message) or `Some(format_args!(..))` (custom message)
                let msg_arg = &rest[2];
                match msg_arg.kind {
                    ExprKind::Call(_, [fmt_arg]) => Self::Format(FormatArgsExpn::parse(cx, fmt_arg)?),
                    _ => Self::Empty,
                }
            },
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
    find_assert_args_inner(cx, expr, expn).map(|([e], mut p)| {
        // `assert!(..)` expands to `core::panicking::panic("assertion failed: ...")` (which we map to
        // `PanicExpn::Str(..)`) and `assert!(.., "..")` expands to
        // `core::panicking::panic_fmt(format_args!(".."))` (which we map to `PanicExpn::Format(..)`).
        // So even we got `PanicExpn::Str(..)` that means there is no custom message provided
        if let PanicExpn::Str(_) = p {
            p = PanicExpn::Empty;
        }

        (e, p)
    })
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
    let panic_expn = for_each_expr(expr, |e| {
        if args.is_full() {
            match PanicExpn::parse(cx, e) {
                Some(expn) => ControlFlow::Break(expn),
                None => ControlFlow::Continue(Descend::Yes),
            }
        } else if is_assert_arg(cx, e, expn) {
            args.push(e);
            ControlFlow::Continue(Descend::No)
        } else {
            ControlFlow::Continue(Descend::Yes)
        }
    });
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
    for_each_expr(expr, |e| {
        if !e.span.from_expansion() {
            return ControlFlow::Continue(Descend::No);
        }
        let e_expn = e.span.ctxt().outer_expn();
        if e_expn == expn {
            ControlFlow::Continue(Descend::Yes)
        } else if e_expn.expn_data().macro_def_id.map(|id| cx.tcx.item_name(id)) == Some(assert_name) {
            ControlFlow::Break((e, e_expn))
        } else {
            ControlFlow::Continue(Descend::No)
        }
    })
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
                sym::cfg => ControlFlow::Continue(()),
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

thread_local! {
    /// We preserve the [`FormatArgs`] structs from the early pass for use in the late pass to be
    /// able to access the many features of a [`LateContext`].
    ///
    /// A thread local is used because [`FormatArgs`] is `!Send` and `!Sync`, we are making an
    /// assumption that the early pass the populates the map and the later late passes will all be
    /// running on the same thread.
    static AST_FORMAT_ARGS: RefCell<FxHashMap<Span, FormatArgs>> = {
        static CALLED: AtomicBool = AtomicBool::new(false);
        debug_assert!(
            !CALLED.swap(true, Ordering::SeqCst),
            "incorrect assumption: `AST_FORMAT_ARGS` should only be accessed by a single thread",
        );

        RefCell::default()
    };
}

/// Record [`rustc_ast::FormatArgs`] for use in late lint passes, this should only be called by
/// `FormatArgsCollector`
pub fn collect_ast_format_args(span: Span, format_args: &FormatArgs) {
    AST_FORMAT_ARGS.with(|ast_format_args| {
        ast_format_args.borrow_mut().insert(span, format_args.clone());
    });
}

/// Calls `callback` with an AST [`FormatArgs`] node if one is found
pub fn find_format_args(cx: &LateContext<'_>, start: &Expr<'_>, expn_id: ExpnId, callback: impl FnOnce(&FormatArgs)) {
    let format_args_expr = for_each_expr(start, |expr| {
        let ctxt = expr.span.ctxt();
        if ctxt == start.span.ctxt() {
            ControlFlow::Continue(Descend::Yes)
        } else if ctxt.outer_expn().is_descendant_of(expn_id)
            && macro_backtrace(expr.span)
                .map(|macro_call| cx.tcx.item_name(macro_call.def_id))
                .any(|name| matches!(name, sym::const_format_args | sym::format_args | sym::format_args_nl))
        {
            ControlFlow::Break(expr)
        } else {
            ControlFlow::Continue(Descend::No)
        }
    });

    if let Some(format_args_expr) = format_args_expr {
        AST_FORMAT_ARGS.with(|ast_format_args| {
            ast_format_args.borrow().get(&format_args_expr.span).map(callback);
        });
    }
}

/// Returns the [`Span`] of the value at `index` extended to the previous comma, e.g. for the value
/// `10`
///
/// ```ignore
/// format("{}.{}", 10, 11)
/// //            ^^^^
/// ```
pub fn format_arg_removal_span(format_args: &FormatArgs, index: usize) -> Option<Span> {
    let ctxt = format_args.span.ctxt();

    let current = hygiene::walk_chain(format_args.arguments.by_index(index)?.expr.span, ctxt);

    let prev = if index == 0 {
        format_args.span
    } else {
        hygiene::walk_chain(format_args.arguments.by_index(index - 1)?.expr.span, ctxt)
    };

    Some(current.with_lo(prev.hi()))
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
        // Sometimes the original string comes from a macro which accepts a malformed string, such as in a
        // #[display(""somestring)] attribute (accepted by the `displaythis` crate). Reconstructing the
        // string from the span will not be possible, so we will just return None here.
        let mut unparsable = false;
        unescape_literal(inner, mode, &mut |_, ch| match ch {
            Ok(ch) => unescaped.push(ch),
            Err(e) if !e.is_fatal() => (),
            Err(_) => unparsable = true,
        });
        if unparsable {
            return None;
        }

        let mut parts = Vec::new();
        let _: Option<!> = for_each_expr(pieces, |expr| {
            if let ExprKind::Lit(lit) = &expr.kind
                && let LitKind::Str(symbol, _) = lit.node
            {
                parts.push(symbol);
            }
            ControlFlow::Continue(())
        });

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
    /// Values passed after the format string and implicit captures. `[1, z + 2, x]` for
    /// `format!("{x} {} {}", 1, z + 2)`.
    value_args: Vec<&'tcx Expr<'tcx>>,
    /// Maps an `rt::v1::Argument::position` or an `rt::v1::Count::Param` to its index in
    /// `value_args`
    pos_to_value_index: Vec<usize>,
    /// Used to check if a value is declared inline & to resolve `InnerSpan`s.
    format_string_span: SpanData,
}

impl<'tcx> FormatArgsValues<'tcx> {
    fn new_empty(format_string_span: SpanData) -> Self {
        Self {
            value_args: Vec::new(),
            pos_to_value_index: Vec::new(),
            format_string_span,
        }
    }

    fn new(args: &'tcx Expr<'tcx>, format_string_span: SpanData) -> Self {
        let mut pos_to_value_index = Vec::new();
        let mut value_args = Vec::new();
        let _: Option<!> = for_each_expr(args, |expr| {
            if expr.span.ctxt() == args.span.ctxt() {
                // ArgumentV1::new_<format_trait>(<val>)
                // ArgumentV1::from_usize(<val>)
                if let ExprKind::Call(callee, [val]) = expr.kind
                    && let ExprKind::Path(QPath::TypeRelative(ty, _)) = callee.kind
                    && let TyKind::Path(QPath::LangItem(LangItem::FormatArgument, _, _)) = ty.kind
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
                ControlFlow::Continue(Descend::Yes)
            } else {
                // assume that any expr with a differing span is a value
                value_args.push(expr);
                ControlFlow::Continue(Descend::No)
            }
        });

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

impl<'tcx> Visitor<'tcx> for ParamPosition {
    fn visit_expr_field(&mut self, field: &'tcx ExprField<'tcx>) {
        match field.ident.name {
            sym::position => {
                if let ExprKind::Lit(lit) = &field.expr.kind
                    && let LitKind::Int(pos, _) = lit.node
                {
                    self.value = pos as usize;
                }
            },
            sym::precision => {
                self.precision = parse_count(field.expr);
            },
            sym::width => {
                self.width = parse_count(field.expr);
            },
            _ => walk_expr(self, field.expr),
        }
    }
}

fn parse_count(expr: &Expr<'_>) -> Option<usize> {
    // <::core::fmt::rt::v1::Count>::Param(1usize),
    if let ExprKind::Call(ctor, [val]) = expr.kind
        && let ExprKind::Path(QPath::TypeRelative(_, path)) = ctor.kind
            && path.ident.name == sym::Param
            && let ExprKind::Lit(lit) = &val.kind
            && let LitKind::Int(pos, _) = lit.node
    {
        Some(pos as usize)
    } else {
        None
    }
}

/// Parses the `fmt` arg of `Arguments::new_v1_formatted(pieces, args, fmt, _)`
fn parse_rt_fmt<'tcx>(fmt_arg: &'tcx Expr<'tcx>) -> Option<impl Iterator<Item = ParamPosition> + 'tcx> {
    if let ExprKind::AddrOf(.., array) = fmt_arg.kind
        && let ExprKind::Array(specs) = array.kind
    {
        Some(specs.iter().map(|spec| {
            if let ExprKind::Call(f, args) = spec.kind
                && let ExprKind::Path(QPath::TypeRelative(ty, f)) = f.kind
                && let TyKind::Path(QPath::LangItem(LangItem::FormatPlaceholder, _, _)) = ty.kind
                && f.ident.name == sym::new
                && let [position, _fill, _align, _flags, precision, width] = args
                && let ExprKind::Lit(position) = &position.kind
                && let LitKind::Int(position, _) = position.node {
                    ParamPosition {
                        value: position as usize,
                        width: parse_count(width),
                        precision: parse_count(precision),
                    }
            } else {
                ParamPosition::default()
            }
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

/// How a format parameter is used in the format string
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FormatParamKind {
    /// An implicit parameter , such as `{}` or `{:?}`.
    Implicit,
    /// A parameter with an explicit number, e.g. `{1}`, `{0:?}`, or `{:.0$}`
    Numbered,
    /// A parameter with an asterisk precision. e.g. `{:.*}`.
    Starred,
    /// A named parameter with a named `value_arg`, such as the `x` in `format!("{x}", x = 1)`.
    Named(Symbol),
    /// An implicit named parameter, such as the `y` in `format!("{y}")`.
    NamedInline(Symbol),
}

/// Where a format parameter is being used in the format string
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FormatParamUsage {
    /// Appears as an argument, e.g. `format!("{}", foo)`
    Argument,
    /// Appears as a width, e.g. `format!("{:width$}", foo, width = 1)`
    Width,
    /// Appears as a precision, e.g. `format!("{:.precision$}", foo, precision = 1)`
    Precision,
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
    /// Where this format param is being used - argument/width/precision
    pub usage: FormatParamUsage,
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
        usage: FormatParamUsage,
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

        Some(Self {
            value,
            kind,
            usage,
            span,
        })
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
    Implied(Option<Span>),
}

impl<'tcx> Count<'tcx> {
    fn new(
        usage: FormatParamUsage,
        count: rpf::Count<'_>,
        position: Option<usize>,
        inner: Option<rpf::InnerSpan>,
        values: &FormatArgsValues<'tcx>,
    ) -> Option<Self> {
        let span = inner.map(|inner| span_from_inner(values.format_string_span, inner));

        Some(match count {
            rpf::Count::CountIs(val) => Self::Is(val, span?),
            rpf::Count::CountIsName(name, _) => Self::Param(FormatParam::new(
                FormatParamKind::Named(Symbol::intern(name)),
                usage,
                position?,
                inner?,
                values,
            )?),
            rpf::Count::CountIsParam(_) => Self::Param(FormatParam::new(
                FormatParamKind::Numbered,
                usage,
                position?,
                inner?,
                values,
            )?),
            rpf::Count::CountIsStar(_) => Self::Param(FormatParam::new(
                FormatParamKind::Starred,
                usage,
                position?,
                inner?,
                values,
            )?),
            rpf::Count::CountImplied => Self::Implied(span),
        })
    }

    pub fn is_implied(self) -> bool {
        matches!(self, Count::Implied(_))
    }

    pub fn param(self) -> Option<FormatParam<'tcx>> {
        match self {
            Count::Param(param) => Some(param),
            _ => None,
        }
    }

    pub fn span(self) -> Option<Span> {
        match self {
            Count::Is(_, span) => Some(span),
            Count::Param(param) => Some(param.span),
            Count::Implied(span) => span,
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
    /// Whether all flag options are set to default (no flags specified).
    pub no_flags: bool,
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
            no_flags: spec.sign.is_none() && !spec.alternate && !spec.zero_pad && spec.debug_hex.is_none(),
            precision: Count::new(
                FormatParamUsage::Precision,
                spec.precision,
                positions.precision,
                spec.precision_span,
                values,
            )?,
            width: Count::new(
                FormatParamUsage::Width,
                spec.width,
                positions.width,
                spec.width_span,
                values,
            )?,
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

    /// Returns true if this format spec is unchanged from the default. e.g. returns true for `{}`,
    /// `{foo}` and `{2}`, but false for `{:?}`, `{foo:5}` and `{3:.5}`
    pub fn is_default(&self) -> bool {
        self.r#trait == sym::Display && self.is_default_for_trait()
    }

    /// Has no other formatting specifiers than setting the format trait. returns true for `{}`,
    /// `{foo}`, `{:?}`, but false for `{foo:5}`, `{3:.5?}`
    pub fn is_default_for_trait(&self) -> bool {
        self.width.is_implied() && self.precision.is_implied() && self.align == Alignment::AlignUnknown && self.no_flags
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

impl<'tcx> FormatArg<'tcx> {
    /// Span of the `:` and format specifiers
    ///
    /// ```ignore
    /// format!("{:.}"), format!("{foo:.}")
    ///           ^^                  ^^
    /// ```
    pub fn format_span(&self) -> Span {
        let base = self.span.data();

        // `base.hi` is `{...}|`, subtract 1 byte (the length of '}') so that it points before the closing
        // brace `{...|}`
        Span::new(self.param.span.hi(), base.hi - BytePos(1), base.ctxt, base.parent)
    }
}

/// A parsed `format_args!` expansion.
#[derive(Debug)]
pub struct FormatArgsExpn<'tcx> {
    /// The format string literal.
    pub format_string: FormatString,
    /// The format arguments, such as `{:?}`.
    pub args: Vec<FormatArg<'tcx>>,
    /// Has an added newline due to `println!()`/`writeln!()`/etc. The last format string part will
    /// include this added newline.
    pub newline: bool,
    /// Spans of the commas between the format string and explicit values, excluding any trailing
    /// comma
    ///
    /// ```ignore
    /// format!("..", 1, 2, 3,)
    /// //          ^  ^  ^
    /// ```
    comma_spans: Vec<Span>,
    /// Explicit values passed after the format string, ignoring implicit captures. `[1, z + 2]` for
    /// `format!("{x} {} {y}", 1, z + 2)`.
    explicit_values: Vec<&'tcx Expr<'tcx>>,
}

impl<'tcx> FormatArgsExpn<'tcx> {
    /// Gets the spans of the commas inbetween the format string and explicit args, not including
    /// any trailing comma
    ///
    /// ```ignore
    /// format!("{} {}", a, b)
    /// //             ^  ^
    /// ```
    ///
    /// Ensures that the format string and values aren't coming from a proc macro that sets the
    /// output span to that of its input
    fn comma_spans(cx: &LateContext<'_>, explicit_values: &[&Expr<'_>], fmt_span: Span) -> Option<Vec<Span>> {
        // `format!("{} {} {c}", "one", "two", c = "three")`
        //                       ^^^^^  ^^^^^      ^^^^^^^
        let value_spans = explicit_values
            .iter()
            .map(|val| hygiene::walk_chain(val.span, fmt_span.ctxt()));

        // `format!("{} {} {c}", "one", "two", c = "three")`
        //                     ^^     ^^     ^^^^^^
        let between_spans = once(fmt_span)
            .chain(value_spans)
            .tuple_windows()
            .map(|(start, end)| start.between(end));

        let mut comma_spans = Vec::new();
        for between_span in between_spans {
            let mut offset = 0;
            let mut seen_comma = false;

            for token in tokenize(&snippet_opt(cx, between_span)?) {
                match token.kind {
                    TokenKind::LineComment { .. } | TokenKind::BlockComment { .. } | TokenKind::Whitespace => {},
                    TokenKind::Comma if !seen_comma => {
                        seen_comma = true;

                        let base = between_span.data();
                        comma_spans.push(Span::new(
                            base.lo + BytePos(offset),
                            base.lo + BytePos(offset + 1),
                            base.ctxt,
                            base.parent,
                        ));
                    },
                    // named arguments, `start_val, name = end_val`
                    //                            ^^^^^^^^^ between_span
                    TokenKind::Ident | TokenKind::Eq if seen_comma => {},
                    // An unexpected token usually indicates the format string or a value came from a proc macro output
                    // that sets the span of its output to an input, e.g. `println!(some_proc_macro!("input"), ..)` that
                    // emits a string literal with the span set to that of `"input"`
                    _ => return None,
                }
                offset += token.len;
            }

            if !seen_comma {
                return None;
            }
        }

        Some(comma_spans)
    }

    pub fn parse(cx: &LateContext<'_>, expr: &'tcx Expr<'tcx>) -> Option<Self> {
        let macro_name = macro_backtrace(expr.span)
            .map(|macro_call| cx.tcx.item_name(macro_call.def_id))
            .find(|&name| matches!(name, sym::const_format_args | sym::format_args | sym::format_args_nl))?;
        let newline = macro_name == sym::format_args_nl;

        // ::core::fmt::Arguments::new_const(pieces)
        // ::core::fmt::Arguments::new_v1(pieces, args)
        // ::core::fmt::Arguments::new_v1_formatted(pieces, args, fmt, _unsafe_arg)
        if let ExprKind::Call(callee, [pieces, rest @ ..]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = callee.kind
            && let TyKind::Path(QPath::LangItem(LangItem::FormatArguments, _, _)) = ty.kind
            && matches!(seg.ident.as_str(), "new_const" | "new_v1" | "new_v1_formatted")
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

            let positions = if let Some(fmt_arg) = rest.get(1) {
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

            let values = if let Some(args) = rest.first() {
                FormatArgsValues::new(args, format_string.span.data())
            } else {
                FormatArgsValues::new_empty(format_string.span.data())
            };

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
                            FormatParamUsage::Argument,
                            position.value,
                            parsed_arg.position_span,
                            &values,
                        )?,
                        format: FormatSpec::new(parsed_arg.format, position, &values)?,
                        span: span_from_inner(values.format_string_span, arg_span),
                    })
                })
                .collect::<Option<Vec<_>>>()?;

            let mut explicit_values = values.value_args;
            // remove values generated for implicitly captured vars
            let len = explicit_values
                .iter()
                .take_while(|val| !format_string.span.contains(val.span))
                .count();
            explicit_values.truncate(len);

            let comma_spans = Self::comma_spans(cx, &explicit_values, format_string.span)?;

            Some(Self {
                format_string,
                args,
                newline,
                comma_spans,
                explicit_values,
            })
        } else {
            None
        }
    }

    pub fn find_nested(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, expn_id: ExpnId) -> Option<Self> {
        for_each_expr(expr, |e| {
            let e_ctxt = e.span.ctxt();
            if e_ctxt == expr.span.ctxt() {
                ControlFlow::Continue(Descend::Yes)
            } else if e_ctxt.outer_expn().is_descendant_of(expn_id) {
                if let Some(args) = FormatArgsExpn::parse(cx, e) {
                    ControlFlow::Break(args)
                } else {
                    ControlFlow::Continue(Descend::No)
                }
            } else {
                ControlFlow::Continue(Descend::No)
            }
        })
    }

    /// Source callsite span of all inputs
    pub fn inputs_span(&self) -> Span {
        match *self.explicit_values {
            [] => self.format_string.span,
            [.., last] => self
                .format_string
                .span
                .to(hygiene::walk_chain(last.span, self.format_string.span.ctxt())),
        }
    }

    /// Get the span of a value expanded to the previous comma, e.g. for the value `10`
    ///
    /// ```ignore
    /// format("{}.{}", 10, 11)
    /// //            ^^^^
    /// ```
    pub fn value_with_prev_comma_span(&self, value_id: HirId) -> Option<Span> {
        for (comma_span, value) in zip(&self.comma_spans, &self.explicit_values) {
            if value.hir_id == value_id {
                return Some(comma_span.to(hygiene::walk_chain(value.span, comma_span.ctxt())));
            }
        }

        None
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
