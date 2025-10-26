#![allow(clippy::similar_names)] // `expr` and `expn`

use std::sync::{Arc, OnceLock};

use crate::visitors::{Descend, for_each_expr_without_closures};
use crate::{get_unique_attr, sym};

use arrayvec::ArrayVec;
use rustc_ast::{FormatArgs, FormatArgument, FormatPlaceholder};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{self as hir, Expr, ExprKind, HirId, Node, QPath};
use rustc_lint::{LateContext, LintContext};
use rustc_span::def_id::DefId;
use rustc_span::hygiene::{self, MacroKind, SyntaxContext};
use rustc_span::{BytePos, ExpnData, ExpnId, ExpnKind, Span, SpanData, Symbol};
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
    sym::todo_macro,
    sym::unimplemented_macro,
    sym::write_macro,
    sym::writeln_macro,
];

/// Returns true if a given Macro `DefId` is a format macro (e.g. `println!`)
pub fn is_format_macro(cx: &LateContext<'_>, macro_def_id: DefId) -> bool {
    if let Some(name) = cx.tcx.get_diagnostic_name(macro_def_id) {
        FORMAT_MACRO_DIAG_ITEMS.contains(&name)
    } else {
        // Allow users to tag any macro as being format!-like
        // TODO: consider deleting FORMAT_MACRO_DIAG_ITEMS and using just this method
        get_unique_attr(cx.sess(), cx.tcx.get_all_attrs(macro_def_id), sym::format_args).is_some()
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
        .is_none_or(DefId::is_local)
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
///
/// If you only want to check whether the root macro has a specific name,
/// consider using [`matching_root_macro_call`] instead.
pub fn root_macro_call(span: Span) -> Option<MacroCall> {
    macro_backtrace(span).last()
}

/// A combination of [`root_macro_call`] and
/// [`is_diagnostic_item`](rustc_middle::ty::TyCtxt::is_diagnostic_item) that returns a `MacroCall`
/// at the root expansion if only it matches the given name.
pub fn matching_root_macro_call(cx: &LateContext<'_>, span: Span, name: Symbol) -> Option<MacroCall> {
    root_macro_call(span).filter(|mc| cx.tcx.is_diagnostic_item(name, mc.def_id))
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
/// macro call site (i.e. the parent of the macro expansion).
///
/// This generally means that `node` is the outermost node of an entire macro expansion, but there
/// are some caveats noted below. This is useful for finding macro calls while visiting the HIR
/// without processing the macro call at every node within its expansion.
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
    let mut parent_iter = cx.tcx.hir_parent_iter(node.hir_id());
    let (parent_id, _) = match parent_iter.next() {
        None => return Some(ExpnId::root()),
        Some((_, Node::Stmt(_))) => match parent_iter.next() {
            None => return Some(ExpnId::root()),
            Some(next) => next,
        },
        Some(next) => next,
    };

    // get the macro expansion of the parent node
    let parent_span = cx.tcx.hir_span(parent_id);
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
    let Some(name) = cx.tcx.get_diagnostic_name(def_id) else {
        return false;
    };
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
    let Some(name) = cx.tcx.get_diagnostic_name(def_id) else {
        return false;
    };
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
    Format(&'a Expr<'a>),
}

impl<'a> PanicExpn<'a> {
    pub fn parse(expr: &'a Expr<'a>) -> Option<Self> {
        let ExprKind::Call(callee, args) = &expr.kind else {
            return None;
        };
        let ExprKind::Path(QPath::Resolved(_, path)) = &callee.kind else {
            return None;
        };
        let name = path.segments.last().unwrap().ident.name;

        let [arg, rest @ ..] = args else {
            return None;
        };
        let result = match name {
            sym::panic if arg.span.eq_ctxt(expr.span) => Self::Empty,
            sym::panic | sym::panic_str => Self::Str(arg),
            sym::panic_display => {
                let ExprKind::AddrOf(_, _, e) = &arg.kind else {
                    return None;
                };
                Self::Display(e)
            },
            sym::panic_fmt => Self::Format(arg),
            // Since Rust 1.52, `assert_{eq,ne}` macros expand to use:
            // `core::panicking::assert_failed(.., left_val, right_val, None | Some(format_args!(..)));`
            sym::assert_failed => {
                // It should have 4 arguments in total (we already matched with the first argument,
                // so we're just checking for 3)
                if rest.len() != 3 {
                    return None;
                }
                // `msg_arg` is either `None` (no custom message) or `Some(format_args!(..))` (custom message)
                let msg_arg = &rest[2];
                match msg_arg.kind {
                    ExprKind::Call(_, [fmt_arg]) => Self::Format(fmt_arg),
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
    let panic_expn = for_each_expr_without_closures(expr, |e| {
        if args.is_full() {
            match PanicExpn::parse(e) {
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
    for_each_expr_without_closures(expr, |e| {
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

/// Stores AST [`FormatArgs`] nodes for use in late lint passes, as they are in a desugared form in
/// the HIR
#[derive(Default, Clone)]
pub struct FormatArgsStorage(Arc<OnceLock<FxHashMap<Span, FormatArgs>>>);

impl FormatArgsStorage {
    /// Returns an AST [`FormatArgs`] node if a `format_args` expansion is found as a descendant of
    /// `expn_id`
    ///
    /// See also [`find_format_arg_expr`]
    pub fn get(&self, cx: &LateContext<'_>, start: &Expr<'_>, expn_id: ExpnId) -> Option<&FormatArgs> {
        let format_args_expr = for_each_expr_without_closures(start, |expr| {
            let ctxt = expr.span.ctxt();
            if ctxt.outer_expn().is_descendant_of(expn_id) {
                if macro_backtrace(expr.span)
                    .map(|macro_call| cx.tcx.item_name(macro_call.def_id))
                    .any(|name| matches!(name, sym::const_format_args | sym::format_args | sym::format_args_nl))
                {
                    ControlFlow::Break(expr)
                } else {
                    ControlFlow::Continue(Descend::Yes)
                }
            } else {
                ControlFlow::Continue(Descend::No)
            }
        })?;

        debug_assert!(self.0.get().is_some(), "`FormatArgsStorage` not yet populated");

        self.0.get()?.get(&format_args_expr.span.with_parent(None))
    }

    /// Should only be called by `FormatArgsCollector`
    pub fn set(&self, format_args: FxHashMap<Span, FormatArgs>) {
        self.0
            .set(format_args)
            .expect("`FormatArgsStorage::set` should only be called once");
    }
}

/// Attempt to find the [`rustc_hir::Expr`] that corresponds to the [`FormatArgument`]'s value
pub fn find_format_arg_expr<'hir>(start: &'hir Expr<'hir>, target: &FormatArgument) -> Option<&'hir Expr<'hir>> {
    let SpanData {
        lo,
        hi,
        ctxt,
        parent: _,
    } = target.expr.span.data();

    for_each_expr_without_closures(start, |expr| {
        // When incremental compilation is enabled spans gain a parent during AST to HIR lowering,
        // since we're comparing an AST span to a HIR one we need to ignore the parent field
        let data = expr.span.data();
        if data.lo == lo && data.hi == hi && data.ctxt == ctxt {
            ControlFlow::Break(expr)
        } else {
            ControlFlow::Continue(())
        }
    })
}

/// Span of the `:` and format specifiers
///
/// ```ignore
/// format!("{:.}"), format!("{foo:.}")
///           ^^                  ^^
/// ```
pub fn format_placeholder_format_span(placeholder: &FormatPlaceholder) -> Option<Span> {
    let base = placeholder.span?.data();

    // `base.hi` is `{...}|`, subtract 1 byte (the length of '}') so that it points before the closing
    // brace `{...|}`
    Some(Span::new(
        placeholder.argument.span?.hi(),
        base.hi - BytePos(1),
        base.ctxt,
        base.parent,
    ))
}

/// Span covering the format string and values
///
/// ```ignore
/// format("{}.{}", 10, 11)
/// //     ^^^^^^^^^^^^^^^
/// ```
pub fn format_args_inputs_span(format_args: &FormatArgs) -> Span {
    match format_args.arguments.explicit_args() {
        [] => format_args.span,
        [.., last] => format_args
            .span
            .to(hygiene::walk_chain(last.expr.span, format_args.span.ctxt())),
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
