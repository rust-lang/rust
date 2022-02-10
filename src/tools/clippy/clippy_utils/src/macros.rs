#![allow(clippy::similar_names)] // `expr` and `expn`

use crate::visitors::expr_visitor_no_bodies;

use arrayvec::ArrayVec;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, Expr, ExprKind, HirId, Node, QPath};
use rustc_lint::LateContext;
use rustc_span::def_id::DefId;
use rustc_span::hygiene::{self, MacroKind, SyntaxContext};
use rustc_span::{sym, ExpnData, ExpnId, ExpnKind, Span, Symbol};
use std::ops::ControlFlow;

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

/// A parsed `format_args!` expansion
#[derive(Debug)]
pub struct FormatArgsExpn<'tcx> {
    /// Span of the first argument, the format string
    pub format_string_span: Span,
    /// The format string split by formatted args like `{..}`
    pub format_string_parts: Vec<Symbol>,
    /// Values passed after the format string
    pub value_args: Vec<&'tcx Expr<'tcx>>,
    /// Each element is a `value_args` index and a formatting trait (e.g. `sym::Debug`)
    pub formatters: Vec<(usize, Symbol)>,
    /// List of `fmt::v1::Argument { .. }` expressions. If this is empty,
    /// then `formatters` represents the format args (`{..}`).
    /// If this is non-empty, it represents the format args, and the `position`
    /// parameters within the struct expressions are indexes of `formatters`.
    pub specs: Vec<&'tcx Expr<'tcx>>,
}

impl<'tcx> FormatArgsExpn<'tcx> {
    /// Parses an expanded `format_args!` or `format_args_nl!` invocation
    pub fn parse(cx: &LateContext<'_>, expr: &'tcx Expr<'tcx>) -> Option<Self> {
        macro_backtrace(expr.span).find(|macro_call| {
            matches!(
                cx.tcx.item_name(macro_call.def_id),
                sym::const_format_args | sym::format_args | sym::format_args_nl
            )
        })?;
        let mut format_string_span: Option<Span> = None;
        let mut format_string_parts: Vec<Symbol> = Vec::new();
        let mut value_args: Vec<&Expr<'_>> = Vec::new();
        let mut formatters: Vec<(usize, Symbol)> = Vec::new();
        let mut specs: Vec<&Expr<'_>> = Vec::new();
        expr_visitor_no_bodies(|e| {
            // if we're still inside of the macro definition...
            if e.span.ctxt() == expr.span.ctxt() {
                // ArgumnetV1::new_<format_trait>(<value>)
                if_chain! {
                    if let ExprKind::Call(callee, [val]) = e.kind;
                    if let ExprKind::Path(QPath::TypeRelative(ty, seg)) = callee.kind;
                    if let hir::TyKind::Path(QPath::Resolved(_, path)) = ty.kind;
                    if path.segments.last().unwrap().ident.name == sym::ArgumentV1;
                    if seg.ident.name.as_str().starts_with("new_");
                    then {
                        let val_idx = if_chain! {
                            if val.span.ctxt() == expr.span.ctxt();
                            if let ExprKind::Field(_, field) = val.kind;
                            if let Ok(idx) = field.name.as_str().parse();
                            then {
                                // tuple index
                                idx
                            } else {
                                // assume the value expression is passed directly
                                formatters.len()
                            }
                        };
                        let fmt_trait = match seg.ident.name.as_str() {
                            "new_display" => "Display",
                            "new_debug" => "Debug",
                            "new_lower_exp" => "LowerExp",
                            "new_upper_exp" => "UpperExp",
                            "new_octal" => "Octal",
                            "new_pointer" => "Pointer",
                            "new_binary" => "Binary",
                            "new_lower_hex" => "LowerHex",
                            "new_upper_hex" => "UpperHex",
                            _ => unreachable!(),
                        };
                        formatters.push((val_idx, Symbol::intern(fmt_trait)));
                    }
                }
                if let ExprKind::Struct(QPath::Resolved(_, path), ..) = e.kind {
                    if path.segments.last().unwrap().ident.name == sym::Argument {
                        specs.push(e);
                    }
                }
                // walk through the macro expansion
                return true;
            }
            // assume that the first expr with a differing context represents
            // (and has the span of) the format string
            if format_string_span.is_none() {
                format_string_span = Some(e.span);
                let span = e.span;
                // walk the expr and collect string literals which are format string parts
                expr_visitor_no_bodies(|e| {
                    if e.span.ctxt() != span.ctxt() {
                        // defensive check, probably doesn't happen
                        return false;
                    }
                    if let ExprKind::Lit(lit) = &e.kind {
                        if let LitKind::Str(symbol, _s) = lit.node {
                            format_string_parts.push(symbol);
                        }
                    }
                    true
                })
                .visit_expr(e);
            } else {
                // assume that any further exprs with a differing context are value args
                value_args.push(e);
            }
            // don't walk anything not from the macro expansion (e.a. inputs)
            false
        })
        .visit_expr(expr);
        Some(FormatArgsExpn {
            format_string_span: format_string_span?,
            format_string_parts,
            value_args,
            formatters,
            specs,
        })
    }

    /// Finds a nested call to `format_args!` within a `format!`-like macro call
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

    /// Returns a vector of `FormatArgsArg`.
    pub fn args(&self) -> Option<Vec<FormatArgsArg<'tcx>>> {
        if self.specs.is_empty() {
            let args = std::iter::zip(&self.value_args, &self.formatters)
                .map(|(value, &(_, format_trait))| FormatArgsArg {
                    value,
                    format_trait,
                    spec: None,
                })
                .collect();
            return Some(args);
        }
        self.specs
            .iter()
            .map(|spec| {
                if_chain! {
                    // struct `core::fmt::rt::v1::Argument`
                    if let ExprKind::Struct(_, fields, _) = spec.kind;
                    if let Some(position_field) = fields.iter().find(|f| f.ident.name == sym::position);
                    if let ExprKind::Lit(lit) = &position_field.expr.kind;
                    if let LitKind::Int(position, _) = lit.node;
                    if let Ok(i) = usize::try_from(position);
                    if let Some(&(j, format_trait)) = self.formatters.get(i);
                    then {
                        Some(FormatArgsArg {
                            value: self.value_args[j],
                            format_trait,
                            spec: Some(spec),
                        })
                    } else {
                        None
                    }
                }
            })
            .collect()
    }

    /// Source callsite span of all inputs
    pub fn inputs_span(&self) -> Span {
        match *self.value_args {
            [] => self.format_string_span,
            [.., last] => self
                .format_string_span
                .to(hygiene::walk_chain(last.span, self.format_string_span.ctxt())),
        }
    }
}

/// Type representing a `FormatArgsExpn`'s format arguments
pub struct FormatArgsArg<'tcx> {
    /// An element of `value_args` according to `position`
    pub value: &'tcx Expr<'tcx>,
    /// An element of `args` according to `position`
    pub format_trait: Symbol,
    /// An element of `specs`
    pub spec: Option<&'tcx Expr<'tcx>>,
}

impl<'tcx> FormatArgsArg<'tcx> {
    /// Returns true if any formatting parameters are used that would have an effect on strings,
    /// like `{:+2}` instead of just `{}`.
    pub fn has_string_formatting(&self) -> bool {
        self.spec.map_or(false, |spec| {
            // `!` because these conditions check that `self` is unformatted.
            !if_chain! {
                // struct `core::fmt::rt::v1::Argument`
                if let ExprKind::Struct(_, fields, _) = spec.kind;
                if let Some(format_field) = fields.iter().find(|f| f.ident.name == sym::format);
                // struct `core::fmt::rt::v1::FormatSpec`
                if let ExprKind::Struct(_, subfields, _) = format_field.expr.kind;
                if subfields.iter().all(|field| match field.ident.name {
                    sym::precision | sym::width => match field.expr.kind {
                        ExprKind::Path(QPath::Resolved(_, path)) => {
                            path.segments.last().unwrap().ident.name == sym::Implied
                        }
                        _ => false,
                    }
                    _ => true,
                });
                then { true } else { false }
            }
        })
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
