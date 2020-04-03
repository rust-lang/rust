#[macro_use]
pub mod sym;

pub mod attrs;
pub mod author;
pub mod camel_case;
pub mod comparisons;
pub mod conf;
pub mod constants;
mod diagnostics;
pub mod higher;
mod hir_utils;
pub mod inspector;
pub mod internal_lints;
pub mod numeric_literal;
pub mod paths;
pub mod ptr;
pub mod sugg;
pub mod usage;
pub use self::attrs::*;
pub use self::diagnostics::*;
pub use self::hir_utils::{SpanlessEq, SpanlessHash};

use std::borrow::Cow;
use std::mem;

use if_chain::if_chain;
use rustc_ast::ast::{self, Attribute, LitKind};
use rustc_attr as attr;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::intravisit::{NestedVisitorMap, Visitor};
use rustc_hir::Node;
use rustc_hir::{
    def, Arm, Block, Body, Constness, Crate, Expr, ExprKind, FnDecl, HirId, ImplItem, ImplItemKind, Item, ItemKind,
    MatchSource, Param, Pat, PatKind, Path, PathSegment, QPath, TraitItem, TraitItemKind, TraitRef, TyKind, Unsafety,
};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, Level, Lint, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::traits;
use rustc_middle::ty::{self, layout::IntegerExt, subst::GenericArg, Binder, Ty, TyCtxt, TypeFoldable};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::source_map::original_sp;
use rustc_span::symbol::{self, kw, Symbol};
use rustc_span::{BytePos, Pos, Span, DUMMY_SP};
use rustc_target::abi::Integer;
use rustc_trait_selection::traits::predicate_for_trait_def;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::query::normalize::AtExt;
use smallvec::SmallVec;

use crate::consts::{constant, Constant};
use crate::reexport::Name;

/// Returns `true` if the two spans come from differing expansions (i.e., one is
/// from a macro and one isn't).
#[must_use]
pub fn differing_macro_contexts(lhs: Span, rhs: Span) -> bool {
    rhs.ctxt() != lhs.ctxt()
}

/// Returns `true` if the given `NodeId` is inside a constant context
///
/// # Example
///
/// ```rust,ignore
/// if in_constant(cx, expr.hir_id) {
///     // Do something
/// }
/// ```
pub fn in_constant(cx: &LateContext<'_, '_>, id: HirId) -> bool {
    let parent_id = cx.tcx.hir().get_parent_item(id);
    match cx.tcx.hir().get(parent_id) {
        Node::Item(&Item {
            kind: ItemKind::Const(..),
            ..
        })
        | Node::TraitItem(&TraitItem {
            kind: TraitItemKind::Const(..),
            ..
        })
        | Node::ImplItem(&ImplItem {
            kind: ImplItemKind::Const(..),
            ..
        })
        | Node::AnonConst(_)
        | Node::Item(&Item {
            kind: ItemKind::Static(..),
            ..
        }) => true,
        Node::Item(&Item {
            kind: ItemKind::Fn(ref sig, ..),
            ..
        })
        | Node::ImplItem(&ImplItem {
            kind: ImplItemKind::Fn(ref sig, _),
            ..
        }) => sig.header.constness == Constness::Const,
        _ => false,
    }
}

/// Returns `true` if this `span` was expanded by any macro.
#[must_use]
pub fn in_macro(span: Span) -> bool {
    if span.from_expansion() {
        if let ExpnKind::Desugaring(..) = span.ctxt().outer_expn_data().kind {
            false
        } else {
            true
        }
    } else {
        false
    }
}
// If the snippet is empty, it's an attribute that was inserted during macro
// expansion and we want to ignore those, because they could come from external
// sources that the user has no control over.
// For some reason these attributes don't have any expansion info on them, so
// we have to check it this way until there is a better way.
pub fn is_present_in_source<T: LintContext>(cx: &T, span: Span) -> bool {
    if let Some(snippet) = snippet_opt(cx, span) {
        if snippet.is_empty() {
            return false;
        }
    }
    true
}

/// Checks if given pattern is a wildcard (`_`)
pub fn is_wild<'tcx>(pat: &impl std::ops::Deref<Target = Pat<'tcx>>) -> bool {
    match pat.kind {
        PatKind::Wild => true,
        _ => false,
    }
}

/// Checks if type is struct, enum or union type with the given def path.
pub fn match_type(cx: &LateContext<'_, '_>, ty: Ty<'_>, path: &[&str]) -> bool {
    match ty.kind {
        ty::Adt(adt, _) => match_def_path(cx, adt.did, path),
        _ => false,
    }
}

/// Checks if the type is equal to a diagnostic item
pub fn is_type_diagnostic_item(cx: &LateContext<'_, '_>, ty: Ty<'_>, diag_item: Symbol) -> bool {
    match ty.kind {
        ty::Adt(adt, _) => cx.tcx.is_diagnostic_item(diag_item, adt.did),
        _ => false,
    }
}

/// Checks if the method call given in `expr` belongs to the given trait.
pub fn match_trait_method(cx: &LateContext<'_, '_>, expr: &Expr<'_>, path: &[&str]) -> bool {
    let def_id = cx.tables.type_dependent_def_id(expr.hir_id).unwrap();
    let trt_id = cx.tcx.trait_of_item(def_id);
    if let Some(trt_id) = trt_id {
        match_def_path(cx, trt_id, path)
    } else {
        false
    }
}

/// Checks if an expression references a variable of the given name.
pub fn match_var(expr: &Expr<'_>, var: Name) -> bool {
    if let ExprKind::Path(QPath::Resolved(None, ref path)) = expr.kind {
        if path.segments.len() == 1 && path.segments[0].ident.name == var {
            return true;
        }
    }
    false
}

pub fn last_path_segment<'tcx>(path: &QPath<'tcx>) -> &'tcx PathSegment<'tcx> {
    match *path {
        QPath::Resolved(_, ref path) => path.segments.last().expect("A path must have at least one segment"),
        QPath::TypeRelative(_, ref seg) => seg,
    }
}

pub fn single_segment_path<'tcx>(path: &QPath<'tcx>) -> Option<&'tcx PathSegment<'tcx>> {
    match *path {
        QPath::Resolved(_, ref path) if path.segments.len() == 1 => Some(&path.segments[0]),
        QPath::Resolved(..) => None,
        QPath::TypeRelative(_, ref seg) => Some(seg),
    }
}

/// Matches a `QPath` against a slice of segment string literals.
///
/// There is also `match_path` if you are dealing with a `rustc_hir::Path` instead of a
/// `rustc_hir::QPath`.
///
/// # Examples
/// ```rust,ignore
/// match_qpath(path, &["std", "rt", "begin_unwind"])
/// ```
pub fn match_qpath(path: &QPath<'_>, segments: &[&str]) -> bool {
    match *path {
        QPath::Resolved(_, ref path) => match_path(path, segments),
        QPath::TypeRelative(ref ty, ref segment) => match ty.kind {
            TyKind::Path(ref inner_path) => {
                !segments.is_empty()
                    && match_qpath(inner_path, &segments[..(segments.len() - 1)])
                    && segment.ident.name.as_str() == segments[segments.len() - 1]
            },
            _ => false,
        },
    }
}

/// Matches a `Path` against a slice of segment string literals.
///
/// There is also `match_qpath` if you are dealing with a `rustc_hir::QPath` instead of a
/// `rustc_hir::Path`.
///
/// # Examples
///
/// ```rust,ignore
/// if match_path(&trait_ref.path, &paths::HASH) {
///     // This is the `std::hash::Hash` trait.
/// }
///
/// if match_path(ty_path, &["rustc", "lint", "Lint"]) {
///     // This is a `rustc_middle::lint::Lint`.
/// }
/// ```
pub fn match_path(path: &Path<'_>, segments: &[&str]) -> bool {
    path.segments
        .iter()
        .rev()
        .zip(segments.iter().rev())
        .all(|(a, b)| a.ident.name.as_str() == *b)
}

/// Matches a `Path` against a slice of segment string literals, e.g.
///
/// # Examples
/// ```rust,ignore
/// match_qpath(path, &["std", "rt", "begin_unwind"])
/// ```
pub fn match_path_ast(path: &ast::Path, segments: &[&str]) -> bool {
    path.segments
        .iter()
        .rev()
        .zip(segments.iter().rev())
        .all(|(a, b)| a.ident.name.as_str() == *b)
}

/// Gets the definition associated to a path.
pub fn path_to_res(cx: &LateContext<'_, '_>, path: &[&str]) -> Option<def::Res> {
    let crates = cx.tcx.crates();
    let krate = crates
        .iter()
        .find(|&&krate| cx.tcx.crate_name(krate).as_str() == path[0]);
    if let Some(krate) = krate {
        let krate = DefId {
            krate: *krate,
            index: CRATE_DEF_INDEX,
        };
        let mut items = cx.tcx.item_children(krate);
        let mut path_it = path.iter().skip(1).peekable();

        loop {
            let segment = match path_it.next() {
                Some(segment) => segment,
                None => return None,
            };

            let result = SmallVec::<[_; 8]>::new();
            for item in mem::replace(&mut items, cx.tcx.arena.alloc_slice(&result)).iter() {
                if item.ident.name.as_str() == *segment {
                    if path_it.peek().is_none() {
                        return Some(item.res);
                    }

                    items = cx.tcx.item_children(item.res.def_id());
                    break;
                }
            }
        }
    } else {
        None
    }
}

pub fn qpath_res(cx: &LateContext<'_, '_>, qpath: &hir::QPath<'_>, id: hir::HirId) -> Res {
    match qpath {
        hir::QPath::Resolved(_, path) => path.res,
        hir::QPath::TypeRelative(..) => {
            if cx.tcx.has_typeck_tables(id.owner.to_def_id()) {
                cx.tcx.typeck_tables_of(id.owner.to_def_id()).qpath_res(qpath, id)
            } else {
                Res::Err
            }
        },
    }
}

/// Convenience function to get the `DefId` of a trait by path.
/// It could be a trait or trait alias.
pub fn get_trait_def_id(cx: &LateContext<'_, '_>, path: &[&str]) -> Option<DefId> {
    let res = match path_to_res(cx, path) {
        Some(res) => res,
        None => return None,
    };

    match res {
        Res::Def(DefKind::Trait | DefKind::TraitAlias, trait_id) => Some(trait_id),
        Res::Err => unreachable!("this trait resolution is impossible: {:?}", &path),
        _ => None,
    }
}

/// Checks whether a type implements a trait.
/// See also `get_trait_def_id`.
pub fn implements_trait<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    ty: Ty<'tcx>,
    trait_id: DefId,
    ty_params: &[GenericArg<'tcx>],
) -> bool {
    let ty = cx.tcx.erase_regions(&ty);
    let obligation = predicate_for_trait_def(
        cx.tcx,
        cx.param_env,
        traits::ObligationCause::dummy(),
        trait_id,
        0,
        ty,
        ty_params,
    );
    cx.tcx
        .infer_ctxt()
        .enter(|infcx| infcx.predicate_must_hold_modulo_regions(&obligation))
}

/// Gets the `hir::TraitRef` of the trait the given method is implemented for.
///
/// Use this if you want to find the `TraitRef` of the `Add` trait in this example:
///
/// ```rust
/// struct Point(isize, isize);
///
/// impl std::ops::Add for Point {
///     type Output = Self;
///
///     fn add(self, other: Self) -> Self {
///         Point(0, 0)
///     }
/// }
/// ```
pub fn trait_ref_of_method<'tcx>(cx: &LateContext<'_, 'tcx>, hir_id: HirId) -> Option<&'tcx TraitRef<'tcx>> {
    // Get the implemented trait for the current function
    let parent_impl = cx.tcx.hir().get_parent_item(hir_id);
    if_chain! {
        if parent_impl != hir::CRATE_HIR_ID;
        if let hir::Node::Item(item) = cx.tcx.hir().get(parent_impl);
        if let hir::ItemKind::Impl{ of_trait: trait_ref, .. } = &item.kind;
        then { return trait_ref.as_ref(); }
    }
    None
}

/// Checks whether this type implements `Drop`.
pub fn has_drop<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.ty_adt_def() {
        Some(def) => def.has_dtor(cx.tcx),
        _ => false,
    }
}

/// Returns the method names and argument list of nested method call expressions that make up
/// `expr`. method/span lists are sorted with the most recent call first.
pub fn method_calls<'tcx>(
    expr: &'tcx Expr<'tcx>,
    max_depth: usize,
) -> (Vec<Symbol>, Vec<&'tcx [Expr<'tcx>]>, Vec<Span>) {
    let mut method_names = Vec::with_capacity(max_depth);
    let mut arg_lists = Vec::with_capacity(max_depth);
    let mut spans = Vec::with_capacity(max_depth);

    let mut current = expr;
    for _ in 0..max_depth {
        if let ExprKind::MethodCall(path, span, args) = &current.kind {
            if args.iter().any(|e| e.span.from_expansion()) {
                break;
            }
            method_names.push(path.ident.name);
            arg_lists.push(&**args);
            spans.push(*span);
            current = &args[0];
        } else {
            break;
        }
    }

    (method_names, arg_lists, spans)
}

/// Matches an `Expr` against a chain of methods, and return the matched `Expr`s.
///
/// For example, if `expr` represents the `.baz()` in `foo.bar().baz()`,
/// `matched_method_chain(expr, &["bar", "baz"])` will return a `Vec`
/// containing the `Expr`s for
/// `.bar()` and `.baz()`
pub fn method_chain_args<'a>(expr: &'a Expr<'_>, methods: &[&str]) -> Option<Vec<&'a [Expr<'a>]>> {
    let mut current = expr;
    let mut matched = Vec::with_capacity(methods.len());
    for method_name in methods.iter().rev() {
        // method chains are stored last -> first
        if let ExprKind::MethodCall(ref path, _, ref args) = current.kind {
            if path.ident.name.as_str() == *method_name {
                if args.iter().any(|e| e.span.from_expansion()) {
                    return None;
                }
                matched.push(&**args); // build up `matched` backwards
                current = &args[0] // go to parent expression
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    // Reverse `matched` so that it is in the same order as `methods`.
    matched.reverse();
    Some(matched)
}

/// Returns `true` if the provided `def_id` is an entrypoint to a program.
pub fn is_entrypoint_fn(cx: &LateContext<'_, '_>, def_id: DefId) -> bool {
    cx.tcx
        .entry_fn(LOCAL_CRATE)
        .map_or(false, |(entry_fn_def_id, _)| def_id == entry_fn_def_id)
}

/// Gets the name of the item the expression is in, if available.
pub fn get_item_name(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> Option<Name> {
    let parent_id = cx.tcx.hir().get_parent_item(expr.hir_id);
    match cx.tcx.hir().find(parent_id) {
        Some(
            Node::Item(Item { ident, .. })
            | Node::TraitItem(TraitItem { ident, .. })
            | Node::ImplItem(ImplItem { ident, .. }),
        ) => Some(ident.name),
        _ => None,
    }
}

/// Gets the name of a `Pat`, if any.
pub fn get_pat_name(pat: &Pat<'_>) -> Option<Name> {
    match pat.kind {
        PatKind::Binding(.., ref spname, _) => Some(spname.name),
        PatKind::Path(ref qpath) => single_segment_path(qpath).map(|ps| ps.ident.name),
        PatKind::Box(ref p) | PatKind::Ref(ref p, _) => get_pat_name(&*p),
        _ => None,
    }
}

struct ContainsName {
    name: Name,
    result: bool,
}

impl<'tcx> Visitor<'tcx> for ContainsName {
    type Map = Map<'tcx>;

    fn visit_name(&mut self, _: Span, name: Name) {
        if self.name == name {
            self.result = true;
        }
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Checks if an `Expr` contains a certain name.
pub fn contains_name(name: Name, expr: &Expr<'_>) -> bool {
    let mut cn = ContainsName { name, result: false };
    cn.visit_expr(expr);
    cn.result
}

/// Converts a span to a code snippet if available, otherwise use default.
///
/// This is useful if you want to provide suggestions for your lint or more generally, if you want
/// to convert a given `Span` to a `str`.
///
/// # Example
/// ```rust,ignore
/// snippet(cx, expr.span, "..")
/// ```
pub fn snippet<'a, T: LintContext>(cx: &T, span: Span, default: &'a str) -> Cow<'a, str> {
    snippet_opt(cx, span).map_or_else(|| Cow::Borrowed(default), From::from)
}

/// Same as `snippet`, but it adapts the applicability level by following rules:
///
/// - Applicability level `Unspecified` will never be changed.
/// - If the span is inside a macro, change the applicability level to `MaybeIncorrect`.
/// - If the default value is used and the applicability level is `MachineApplicable`, change it to
/// `HasPlaceholders`
pub fn snippet_with_applicability<'a, T: LintContext>(
    cx: &T,
    span: Span,
    default: &'a str,
    applicability: &mut Applicability,
) -> Cow<'a, str> {
    if *applicability != Applicability::Unspecified && span.from_expansion() {
        *applicability = Applicability::MaybeIncorrect;
    }
    snippet_opt(cx, span).map_or_else(
        || {
            if *applicability == Applicability::MachineApplicable {
                *applicability = Applicability::HasPlaceholders;
            }
            Cow::Borrowed(default)
        },
        From::from,
    )
}

/// Same as `snippet`, but should only be used when it's clear that the input span is
/// not a macro argument.
pub fn snippet_with_macro_callsite<'a, T: LintContext>(cx: &T, span: Span, default: &'a str) -> Cow<'a, str> {
    snippet(cx, span.source_callsite(), default)
}

/// Converts a span to a code snippet. Returns `None` if not available.
pub fn snippet_opt<T: LintContext>(cx: &T, span: Span) -> Option<String> {
    cx.sess().source_map().span_to_snippet(span).ok()
}

/// Converts a span (from a block) to a code snippet if available, otherwise use default.
///
/// This trims the code of indentation, except for the first line. Use it for blocks or block-like
/// things which need to be printed as such.
///
/// The `indent_relative_to` arg can be used, to provide a span, where the indentation of the
/// resulting snippet of the given span.
///
/// # Example
///
/// ```rust,ignore
/// snippet_block(cx, block.span, "..", None)
/// // where, `block` is the block of the if expr
///     if x {
///         y;
///     }
/// // will return the snippet
/// {
///     y;
/// }
/// ```
///
/// ```rust,ignore
/// snippet_block(cx, block.span, "..", Some(if_expr.span))
/// // where, `block` is the block of the if expr
///     if x {
///         y;
///     }
/// // will return the snippet
/// {
///         y;
///     } // aligned with `if`
/// ```
/// Note that the first line of the snippet always has 0 indentation.
pub fn snippet_block<'a, T: LintContext>(
    cx: &T,
    span: Span,
    default: &'a str,
    indent_relative_to: Option<Span>,
) -> Cow<'a, str> {
    let snip = snippet(cx, span, default);
    let indent = indent_relative_to.and_then(|s| indent_of(cx, s));
    trim_multiline(snip, true, indent)
}

/// Same as `snippet_block`, but adapts the applicability level by the rules of
/// `snippet_with_applicabiliy`.
pub fn snippet_block_with_applicability<'a, T: LintContext>(
    cx: &T,
    span: Span,
    default: &'a str,
    indent_relative_to: Option<Span>,
    applicability: &mut Applicability,
) -> Cow<'a, str> {
    let snip = snippet_with_applicability(cx, span, default, applicability);
    let indent = indent_relative_to.and_then(|s| indent_of(cx, s));
    trim_multiline(snip, true, indent)
}

/// Returns a new Span that extends the original Span to the first non-whitespace char of the first
/// line.
///
/// ```rust,ignore
///     let x = ();
/// //          ^^
/// // will be converted to
///     let x = ();
/// //  ^^^^^^^^^^
/// ```
pub fn first_line_of_span<T: LintContext>(cx: &T, span: Span) -> Span {
    if let Some(first_char_pos) = first_char_in_first_line(cx, span) {
        span.with_lo(first_char_pos)
    } else {
        span
    }
}

fn first_char_in_first_line<T: LintContext>(cx: &T, span: Span) -> Option<BytePos> {
    let line_span = line_span(cx, span);
    if let Some(snip) = snippet_opt(cx, line_span) {
        snip.find(|c: char| !c.is_whitespace())
            .map(|pos| line_span.lo() + BytePos::from_usize(pos))
    } else {
        None
    }
}

/// Returns the indentation of the line of a span
///
/// ```rust,ignore
/// let x = ();
/// //      ^^ -- will return 0
///     let x = ();
/// //          ^^ -- will return 4
/// ```
pub fn indent_of<T: LintContext>(cx: &T, span: Span) -> Option<usize> {
    if let Some(snip) = snippet_opt(cx, line_span(cx, span)) {
        snip.find(|c: char| !c.is_whitespace())
    } else {
        None
    }
}

/// Extends the span to the beginning of the spans line, incl. whitespaces.
///
/// ```rust,ignore
///        let x = ();
/// //             ^^
/// // will be converted to
///        let x = ();
/// // ^^^^^^^^^^^^^^
/// ```
fn line_span<T: LintContext>(cx: &T, span: Span) -> Span {
    let span = original_sp(span, DUMMY_SP);
    let source_map_and_line = cx.sess().source_map().lookup_line(span.lo()).unwrap();
    let line_no = source_map_and_line.line;
    let line_start = source_map_and_line.sf.lines[line_no];
    Span::new(line_start, span.hi(), span.ctxt())
}

/// Like `snippet_block`, but add braces if the expr is not an `ExprKind::Block`.
/// Also takes an `Option<String>` which can be put inside the braces.
pub fn expr_block<'a, T: LintContext>(
    cx: &T,
    expr: &Expr<'_>,
    option: Option<String>,
    default: &'a str,
    indent_relative_to: Option<Span>,
) -> Cow<'a, str> {
    let code = snippet_block(cx, expr.span, default, indent_relative_to);
    let string = option.unwrap_or_default();
    if expr.span.from_expansion() {
        Cow::Owned(format!("{{ {} }}", snippet_with_macro_callsite(cx, expr.span, default)))
    } else if let ExprKind::Block(_, _) = expr.kind {
        Cow::Owned(format!("{}{}", code, string))
    } else if string.is_empty() {
        Cow::Owned(format!("{{ {} }}", code))
    } else {
        Cow::Owned(format!("{{\n{};\n{}\n}}", code, string))
    }
}

/// Trim indentation from a multiline string with possibility of ignoring the
/// first line.
fn trim_multiline(s: Cow<'_, str>, ignore_first: bool, indent: Option<usize>) -> Cow<'_, str> {
    let s_space = trim_multiline_inner(s, ignore_first, indent, ' ');
    let s_tab = trim_multiline_inner(s_space, ignore_first, indent, '\t');
    trim_multiline_inner(s_tab, ignore_first, indent, ' ')
}

fn trim_multiline_inner(s: Cow<'_, str>, ignore_first: bool, indent: Option<usize>, ch: char) -> Cow<'_, str> {
    let mut x = s
        .lines()
        .skip(ignore_first as usize)
        .filter_map(|l| {
            if l.is_empty() {
                None
            } else {
                // ignore empty lines
                Some(l.char_indices().find(|&(_, x)| x != ch).unwrap_or((l.len(), ch)).0)
            }
        })
        .min()
        .unwrap_or(0);
    if let Some(indent) = indent {
        x = x.saturating_sub(indent);
    }
    if x > 0 {
        Cow::Owned(
            s.lines()
                .enumerate()
                .map(|(i, l)| {
                    if (ignore_first && i == 0) || l.is_empty() {
                        l
                    } else {
                        l.split_at(x).1
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        s
    }
}

/// Gets the parent expression, if any –- this is useful to constrain a lint.
pub fn get_parent_expr<'c>(cx: &'c LateContext<'_, '_>, e: &Expr<'_>) -> Option<&'c Expr<'c>> {
    let map = &cx.tcx.hir();
    let hir_id = e.hir_id;
    let parent_id = map.get_parent_node(hir_id);
    if hir_id == parent_id {
        return None;
    }
    map.find(parent_id).and_then(|node| {
        if let Node::Expr(parent) = node {
            Some(parent)
        } else {
            None
        }
    })
}

pub fn get_enclosing_block<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, hir_id: HirId) -> Option<&'tcx Block<'tcx>> {
    let map = &cx.tcx.hir();
    let enclosing_node = map
        .get_enclosing_scope(hir_id)
        .and_then(|enclosing_id| map.find(enclosing_id));
    if let Some(node) = enclosing_node {
        match node {
            Node::Block(block) => Some(block),
            Node::Item(&Item {
                kind: ItemKind::Fn(_, _, eid),
                ..
            })
            | Node::ImplItem(&ImplItem {
                kind: ImplItemKind::Fn(_, eid),
                ..
            }) => match cx.tcx.hir().body(eid).value.kind {
                ExprKind::Block(ref block, _) => Some(block),
                _ => None,
            },
            _ => None,
        }
    } else {
        None
    }
}

/// Returns the base type for HIR references and pointers.
pub fn walk_ptrs_hir_ty<'tcx>(ty: &'tcx hir::Ty<'tcx>) -> &'tcx hir::Ty<'tcx> {
    match ty.kind {
        TyKind::Ptr(ref mut_ty) | TyKind::Rptr(_, ref mut_ty) => walk_ptrs_hir_ty(&mut_ty.ty),
        _ => ty,
    }
}

/// Returns the base type for references and raw pointers.
pub fn walk_ptrs_ty(ty: Ty<'_>) -> Ty<'_> {
    match ty.kind {
        ty::Ref(_, ty, _) => walk_ptrs_ty(ty),
        _ => ty,
    }
}

/// Returns the base type for references and raw pointers, and count reference
/// depth.
pub fn walk_ptrs_ty_depth(ty: Ty<'_>) -> (Ty<'_>, usize) {
    fn inner(ty: Ty<'_>, depth: usize) -> (Ty<'_>, usize) {
        match ty.kind {
            ty::Ref(_, ty, _) => inner(ty, depth + 1),
            _ => (ty, depth),
        }
    }
    inner(ty, 0)
}

/// Checks whether the given expression is a constant integer of the given value.
/// unlike `is_integer_literal`, this version does const folding
pub fn is_integer_const(cx: &LateContext<'_, '_>, e: &Expr<'_>, value: u128) -> bool {
    if is_integer_literal(e, value) {
        return true;
    }
    let map = cx.tcx.hir();
    let parent_item = map.get_parent_item(e.hir_id);
    if let Some((Constant::Int(v), _)) = map
        .maybe_body_owned_by(parent_item)
        .and_then(|body_id| constant(cx, cx.tcx.body_tables(body_id), e))
    {
        value == v
    } else {
        false
    }
}

/// Checks whether the given expression is a constant literal of the given value.
pub fn is_integer_literal(expr: &Expr<'_>, value: u128) -> bool {
    // FIXME: use constant folding
    if let ExprKind::Lit(ref spanned) = expr.kind {
        if let LitKind::Int(v, _) = spanned.node {
            return v == value;
        }
    }
    false
}

/// Returns `true` if the given `Expr` has been coerced before.
///
/// Examples of coercions can be found in the Nomicon at
/// <https://doc.rust-lang.org/nomicon/coercions.html>.
///
/// See `rustc_middle::ty::adjustment::Adjustment` and `rustc_typeck::check::coercion` for more
/// information on adjustments and coercions.
pub fn is_adjusted(cx: &LateContext<'_, '_>, e: &Expr<'_>) -> bool {
    cx.tables.adjustments().get(e.hir_id).is_some()
}

/// Returns the pre-expansion span if is this comes from an expansion of the
/// macro `name`.
/// See also `is_direct_expn_of`.
#[must_use]
pub fn is_expn_of(mut span: Span, name: &str) -> Option<Span> {
    loop {
        if span.from_expansion() {
            let data = span.ctxt().outer_expn_data();
            let new_span = data.call_site;

            if let ExpnKind::Macro(MacroKind::Bang, mac_name) = data.kind {
                if mac_name.as_str() == name {
                    return Some(new_span);
                }
            }

            span = new_span;
        } else {
            return None;
        }
    }
}

/// Returns the pre-expansion span if the span directly comes from an expansion
/// of the macro `name`.
/// The difference with `is_expn_of` is that in
/// ```rust,ignore
/// foo!(bar!(42));
/// ```
/// `42` is considered expanded from `foo!` and `bar!` by `is_expn_of` but only
/// `bar!` by
/// `is_direct_expn_of`.
#[must_use]
pub fn is_direct_expn_of(span: Span, name: &str) -> Option<Span> {
    if span.from_expansion() {
        let data = span.ctxt().outer_expn_data();
        let new_span = data.call_site;

        if let ExpnKind::Macro(MacroKind::Bang, mac_name) = data.kind {
            if mac_name.as_str() == name {
                return Some(new_span);
            }
        }
    }

    None
}

/// Convenience function to get the return type of a function.
pub fn return_ty<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, fn_item: hir::HirId) -> Ty<'tcx> {
    let fn_def_id = cx.tcx.hir().local_def_id(fn_item);
    let ret_ty = cx.tcx.fn_sig(fn_def_id).output();
    cx.tcx.erase_late_bound_regions(&ret_ty)
}

/// Checks if two types are the same.
///
/// This discards any lifetime annotations, too.
//
// FIXME: this works correctly for lifetimes bounds (`for <'a> Foo<'a>` ==
// `for <'b> Foo<'b>`, but not for type parameters).
pub fn same_tys<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
    let a = cx.tcx.erase_late_bound_regions(&Binder::bind(a));
    let b = cx.tcx.erase_late_bound_regions(&Binder::bind(b));
    cx.tcx
        .infer_ctxt()
        .enter(|infcx| infcx.can_eq(cx.param_env, a, b).is_ok())
}

/// Returns `true` if the given type is an `unsafe` function.
pub fn type_is_unsafe_function<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind {
        ty::FnDef(..) | ty::FnPtr(_) => ty.fn_sig(cx.tcx).unsafety() == Unsafety::Unsafe,
        _ => false,
    }
}

pub fn is_copy<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.is_copy_modulo_regions(cx.tcx, cx.param_env, DUMMY_SP)
}

/// Checks if an expression is constructing a tuple-like enum variant or struct
pub fn is_ctor_or_promotable_const_function(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::Call(ref fun, _) = expr.kind {
        if let ExprKind::Path(ref qp) = fun.kind {
            let res = cx.tables.qpath_res(qp, fun.hir_id);
            return match res {
                def::Res::Def(DefKind::Variant | DefKind::Ctor(..), ..) => true,
                def::Res::Def(_, def_id) => cx.tcx.is_promotable_const_fn(def_id),
                _ => false,
            };
        }
    }
    false
}

/// Returns `true` if a pattern is refutable.
pub fn is_refutable(cx: &LateContext<'_, '_>, pat: &Pat<'_>) -> bool {
    fn is_enum_variant(cx: &LateContext<'_, '_>, qpath: &QPath<'_>, id: HirId) -> bool {
        matches!(
            cx.tables.qpath_res(qpath, id),
            def::Res::Def(DefKind::Variant, ..) | Res::Def(DefKind::Ctor(def::CtorOf::Variant, _), _)
        )
    }

    fn are_refutable<'a, I: Iterator<Item = &'a Pat<'a>>>(cx: &LateContext<'_, '_>, mut i: I) -> bool {
        i.any(|pat| is_refutable(cx, pat))
    }

    match pat.kind {
        PatKind::Binding(..) | PatKind::Wild => false,
        PatKind::Box(ref pat) | PatKind::Ref(ref pat, _) => is_refutable(cx, pat),
        PatKind::Lit(..) | PatKind::Range(..) => true,
        PatKind::Path(ref qpath) => is_enum_variant(cx, qpath, pat.hir_id),
        PatKind::Or(ref pats) | PatKind::Tuple(ref pats, _) => are_refutable(cx, pats.iter().map(|pat| &**pat)),
        PatKind::Struct(ref qpath, ref fields, _) => {
            if is_enum_variant(cx, qpath, pat.hir_id) {
                true
            } else {
                are_refutable(cx, fields.iter().map(|field| &*field.pat))
            }
        },
        PatKind::TupleStruct(ref qpath, ref pats, _) => {
            if is_enum_variant(cx, qpath, pat.hir_id) {
                true
            } else {
                are_refutable(cx, pats.iter().map(|pat| &**pat))
            }
        },
        PatKind::Slice(ref head, ref middle, ref tail) => {
            are_refutable(cx, head.iter().chain(middle).chain(tail.iter()).map(|pat| &**pat))
        },
    }
}

/// Checks for the `#[automatically_derived]` attribute all `#[derive]`d
/// implementations have.
pub fn is_automatically_derived(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, sym!(automatically_derived))
}

/// Remove blocks around an expression.
///
/// Ie. `x`, `{ x }` and `{{{{ x }}}}` all give `x`. `{ x; y }` and `{}` return
/// themselves.
pub fn remove_blocks<'tcx>(mut expr: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    while let ExprKind::Block(ref block, ..) = expr.kind {
        match (block.stmts.is_empty(), block.expr.as_ref()) {
            (true, Some(e)) => expr = e,
            _ => break,
        }
    }
    expr
}

pub fn is_self(slf: &Param<'_>) -> bool {
    if let PatKind::Binding(.., name, _) = slf.pat.kind {
        name.name == kw::SelfLower
    } else {
        false
    }
}

pub fn is_self_ty(slf: &hir::Ty<'_>) -> bool {
    if_chain! {
        if let TyKind::Path(ref qp) = slf.kind;
        if let QPath::Resolved(None, ref path) = *qp;
        if let Res::SelfTy(..) = path.res;
        then {
            return true
        }
    }
    false
}

pub fn iter_input_pats<'tcx>(decl: &FnDecl<'_>, body: &'tcx Body<'_>) -> impl Iterator<Item = &'tcx Param<'tcx>> {
    (0..decl.inputs.len()).map(move |i| &body.params[i])
}

/// Checks if a given expression is a match expression expanded from the `?`
/// operator or the `try` macro.
pub fn is_try<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    fn is_ok(arm: &Arm<'_>) -> bool {
        if_chain! {
            if let PatKind::TupleStruct(ref path, ref pat, None) = arm.pat.kind;
            if match_qpath(path, &paths::RESULT_OK[1..]);
            if let PatKind::Binding(_, hir_id, _, None) = pat[0].kind;
            if let ExprKind::Path(QPath::Resolved(None, ref path)) = arm.body.kind;
            if let Res::Local(lid) = path.res;
            if lid == hir_id;
            then {
                return true;
            }
        }
        false
    }

    fn is_err(arm: &Arm<'_>) -> bool {
        if let PatKind::TupleStruct(ref path, _, _) = arm.pat.kind {
            match_qpath(path, &paths::RESULT_ERR[1..])
        } else {
            false
        }
    }

    if let ExprKind::Match(_, ref arms, ref source) = expr.kind {
        // desugared from a `?` operator
        if let MatchSource::TryDesugar = *source {
            return Some(expr);
        }

        if_chain! {
            if arms.len() == 2;
            if arms[0].guard.is_none();
            if arms[1].guard.is_none();
            if (is_ok(&arms[0]) && is_err(&arms[1])) ||
                (is_ok(&arms[1]) && is_err(&arms[0]));
            then {
                return Some(expr);
            }
        }
    }

    None
}

/// Returns `true` if the lint is allowed in the current context
///
/// Useful for skipping long running code when it's unnecessary
pub fn is_allowed(cx: &LateContext<'_, '_>, lint: &'static Lint, id: HirId) -> bool {
    cx.tcx.lint_level_at_node(lint, id).0 == Level::Allow
}

pub fn get_arg_name(pat: &Pat<'_>) -> Option<ast::Name> {
    match pat.kind {
        PatKind::Binding(.., ident, None) => Some(ident.name),
        PatKind::Ref(ref subpat, _) => get_arg_name(subpat),
        _ => None,
    }
}

pub fn int_bits(tcx: TyCtxt<'_>, ity: ast::IntTy) -> u64 {
    Integer::from_attr(&tcx, attr::IntType::SignedInt(ity)).size().bits()
}

#[allow(clippy::cast_possible_wrap)]
/// Turn a constant int byte representation into an i128
pub fn sext(tcx: TyCtxt<'_>, u: u128, ity: ast::IntTy) -> i128 {
    let amt = 128 - int_bits(tcx, ity);
    ((u as i128) << amt) >> amt
}

#[allow(clippy::cast_sign_loss)]
/// clip unused bytes
pub fn unsext(tcx: TyCtxt<'_>, u: i128, ity: ast::IntTy) -> u128 {
    let amt = 128 - int_bits(tcx, ity);
    ((u as u128) << amt) >> amt
}

/// clip unused bytes
pub fn clip(tcx: TyCtxt<'_>, u: u128, ity: ast::UintTy) -> u128 {
    let bits = Integer::from_attr(&tcx, attr::IntType::UnsignedInt(ity)).size().bits();
    let amt = 128 - bits;
    (u << amt) >> amt
}

/// Removes block comments from the given `Vec` of lines.
///
/// # Examples
///
/// ```rust,ignore
/// without_block_comments(vec!["/*", "foo", "*/"]);
/// // => vec![]
///
/// without_block_comments(vec!["bar", "/*", "foo", "*/"]);
/// // => vec!["bar"]
/// ```
pub fn without_block_comments(lines: Vec<&str>) -> Vec<&str> {
    let mut without = vec![];

    let mut nest_level = 0;

    for line in lines {
        if line.contains("/*") {
            nest_level += 1;
            continue;
        } else if line.contains("*/") {
            nest_level -= 1;
            continue;
        }

        if nest_level == 0 {
            without.push(line);
        }
    }

    without
}

pub fn any_parent_is_automatically_derived(tcx: TyCtxt<'_>, node: HirId) -> bool {
    let map = &tcx.hir();
    let mut prev_enclosing_node = None;
    let mut enclosing_node = node;
    while Some(enclosing_node) != prev_enclosing_node {
        if is_automatically_derived(map.attrs(enclosing_node)) {
            return true;
        }
        prev_enclosing_node = Some(enclosing_node);
        enclosing_node = map.get_parent_item(enclosing_node);
    }
    false
}

/// Returns true if ty has `iter` or `iter_mut` methods
pub fn has_iter_method(cx: &LateContext<'_, '_>, probably_ref_ty: Ty<'_>) -> Option<&'static str> {
    // FIXME: instead of this hard-coded list, we should check if `<adt>::iter`
    // exists and has the desired signature. Unfortunately FnCtxt is not exported
    // so we can't use its `lookup_method` method.
    let into_iter_collections: [&[&str]; 13] = [
        &paths::VEC,
        &paths::OPTION,
        &paths::RESULT,
        &paths::BTREESET,
        &paths::BTREEMAP,
        &paths::VEC_DEQUE,
        &paths::LINKED_LIST,
        &paths::BINARY_HEAP,
        &paths::HASHSET,
        &paths::HASHMAP,
        &paths::PATH_BUF,
        &paths::PATH,
        &paths::RECEIVER,
    ];

    let ty_to_check = match probably_ref_ty.kind {
        ty::Ref(_, ty_to_check, _) => ty_to_check,
        _ => probably_ref_ty,
    };

    let def_id = match ty_to_check.kind {
        ty::Array(..) => return Some("array"),
        ty::Slice(..) => return Some("slice"),
        ty::Adt(adt, _) => adt.did,
        _ => return None,
    };

    for path in &into_iter_collections {
        if match_def_path(cx, def_id, path) {
            return Some(*path.last().unwrap());
        }
    }
    None
}

/// Matches a function call with the given path and returns the arguments.
///
/// Usage:
///
/// ```rust,ignore
/// if let Some(args) = match_function_call(cx, begin_panic_call, &paths::BEGIN_PANIC);
/// ```
pub fn match_function_call<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    expr: &'tcx Expr<'_>,
    path: &[&str],
) -> Option<&'tcx [Expr<'tcx>]> {
    if_chain! {
        if let ExprKind::Call(ref fun, ref args) = expr.kind;
        if let ExprKind::Path(ref qpath) = fun.kind;
        if let Some(fun_def_id) = cx.tables.qpath_res(qpath, fun.hir_id).opt_def_id();
        if match_def_path(cx, fun_def_id, path);
        then {
            return Some(&args)
        }
    };
    None
}

/// Checks if `Ty` is normalizable. This function is useful
/// to avoid crashes on `layout_of`.
pub fn is_normalizable<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, param_env: ty::ParamEnv<'tcx>, ty: Ty<'tcx>) -> bool {
    cx.tcx.infer_ctxt().enter(|infcx| {
        let cause = rustc_middle::traits::ObligationCause::dummy();
        infcx.at(&cause, param_env).normalize(&ty).is_ok()
    })
}

pub fn match_def_path<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, did: DefId, syms: &[&str]) -> bool {
    let path = cx.get_def_path(did);
    path.len() == syms.len() && path.into_iter().zip(syms.iter()).all(|(a, &b)| a.as_str() == b)
}

/// Returns the list of condition expressions and the list of blocks in a
/// sequence of `if/else`.
/// E.g., this returns `([a, b], [c, d, e])` for the expression
/// `if a { c } else if b { d } else { e }`.
pub fn if_sequence<'tcx>(
    mut expr: &'tcx Expr<'tcx>,
) -> (SmallVec<[&'tcx Expr<'tcx>; 1]>, SmallVec<[&'tcx Block<'tcx>; 1]>) {
    let mut conds = SmallVec::new();
    let mut blocks: SmallVec<[&Block<'_>; 1]> = SmallVec::new();

    while let Some((ref cond, ref then_expr, ref else_expr)) = higher::if_block(&expr) {
        conds.push(&**cond);
        if let ExprKind::Block(ref block, _) = then_expr.kind {
            blocks.push(block);
        } else {
            panic!("ExprKind::If node is not an ExprKind::Block");
        }

        if let Some(ref else_expr) = *else_expr {
            expr = else_expr;
        } else {
            break;
        }
    }

    // final `else {..}`
    if !blocks.is_empty() {
        if let ExprKind::Block(ref block, _) = expr.kind {
            blocks.push(&**block);
        }
    }

    (conds, blocks)
}

pub fn parent_node_is_if_expr<'a, 'b>(expr: &Expr<'_>, cx: &LateContext<'a, 'b>) -> bool {
    let map = cx.tcx.hir();
    let parent_id = map.get_parent_node(expr.hir_id);
    let parent_node = map.get(parent_id);

    match parent_node {
        Node::Expr(e) => higher::if_block(&e).is_some(),
        Node::Arm(e) => higher::if_block(&e.body).is_some(),
        _ => false,
    }
}

// Finds the attribute with the given name, if any
pub fn attr_by_name<'a>(attrs: &'a [Attribute], name: &'_ str) -> Option<&'a Attribute> {
    attrs
        .iter()
        .find(|attr| attr.ident().map_or(false, |ident| ident.as_str() == name))
}

// Finds the `#[must_use]` attribute, if any
pub fn must_use_attr(attrs: &[Attribute]) -> Option<&Attribute> {
    attr_by_name(attrs, "must_use")
}

// Returns whether the type has #[must_use] attribute
pub fn is_must_use_ty<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind {
        ty::Adt(ref adt, _) => must_use_attr(&cx.tcx.get_attrs(adt.did)).is_some(),
        ty::Foreign(ref did) => must_use_attr(&cx.tcx.get_attrs(*did)).is_some(),
        ty::Slice(ref ty)
        | ty::Array(ref ty, _)
        | ty::RawPtr(ty::TypeAndMut { ref ty, .. })
        | ty::Ref(_, ref ty, _) => {
            // for the Array case we don't need to care for the len == 0 case
            // because we don't want to lint functions returning empty arrays
            is_must_use_ty(cx, *ty)
        },
        ty::Tuple(ref substs) => substs.types().any(|ty| is_must_use_ty(cx, ty)),
        ty::Opaque(ref def_id, _) => {
            for (predicate, _) in cx.tcx.predicates_of(*def_id).predicates {
                if let ty::Predicate::Trait(ref poly_trait_predicate, _) = predicate {
                    if must_use_attr(&cx.tcx.get_attrs(poly_trait_predicate.skip_binder().trait_ref.def_id)).is_some() {
                        return true;
                    }
                }
            }
            false
        },
        ty::Dynamic(binder, _) => {
            for predicate in binder.skip_binder().iter() {
                if let ty::ExistentialPredicate::Trait(ref trait_ref) = predicate {
                    if must_use_attr(&cx.tcx.get_attrs(trait_ref.def_id)).is_some() {
                        return true;
                    }
                }
            }
            false
        },
        _ => false,
    }
}

// check if expr is calling method or function with #[must_use] attribyte
pub fn is_must_use_func_call(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> bool {
    let did = match expr.kind {
        ExprKind::Call(ref path, _) => if_chain! {
            if let ExprKind::Path(ref qpath) = path.kind;
            if let def::Res::Def(_, did) = cx.tables.qpath_res(qpath, path.hir_id);
            then {
                Some(did)
            } else {
                None
            }
        },
        ExprKind::MethodCall(_, _, _) => cx.tables.type_dependent_def_id(expr.hir_id),
        _ => None,
    };

    if let Some(did) = did {
        must_use_attr(&cx.tcx.get_attrs(did)).is_some()
    } else {
        false
    }
}

pub fn is_no_std_crate(krate: &Crate<'_>) -> bool {
    krate.item.attrs.iter().any(|attr| {
        if let ast::AttrKind::Normal(ref attr) = attr.kind {
            attr.path == symbol::sym::no_std
        } else {
            false
        }
    })
}

/// Check if parent of a hir node is a trait implementation block.
/// For example, `f` in
/// ```rust,ignore
/// impl Trait for S {
///     fn f() {}
/// }
/// ```
pub fn is_trait_impl_item(cx: &LateContext<'_, '_>, hir_id: HirId) -> bool {
    if let Some(Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(hir_id)) {
        matches!(item.kind, ItemKind::Impl{ of_trait: Some(_), .. })
    } else {
        false
    }
}

/// Check if it's even possible to satisfy the `where` clause for the item.
///
/// `trivial_bounds` feature allows functions with unsatisfiable bounds, for example:
///
/// ```ignore
/// fn foo() where i32: Iterator {
///     for _ in 2i32 {}
/// }
/// ```
pub fn fn_has_unsatisfiable_preds(cx: &LateContext<'_, '_>, did: DefId) -> bool {
    use rustc_trait_selection::traits;
    let predicates = cx
        .tcx
        .predicates_of(did)
        .predicates
        .iter()
        .filter_map(|(p, _)| if p.is_global() { Some(*p) } else { None })
        .collect();
    !traits::normalize_and_test_predicates(cx.tcx, traits::elaborate_predicates(cx.tcx, predicates).collect())
}

#[cfg(test)]
mod test {
    use super::{trim_multiline, without_block_comments};

    #[test]
    fn test_trim_multiline_single_line() {
        assert_eq!("", trim_multiline("".into(), false, None));
        assert_eq!("...", trim_multiline("...".into(), false, None));
        assert_eq!("...", trim_multiline("    ...".into(), false, None));
        assert_eq!("...", trim_multiline("\t...".into(), false, None));
        assert_eq!("...", trim_multiline("\t\t...".into(), false, None));
    }

    #[test]
    #[rustfmt::skip]
    fn test_trim_multiline_block() {
        assert_eq!("\
    if x {
        y
    } else {
        z
    }", trim_multiline("    if x {
            y
        } else {
            z
        }".into(), false, None));
        assert_eq!("\
    if x {
    \ty
    } else {
    \tz
    }", trim_multiline("    if x {
        \ty
        } else {
        \tz
        }".into(), false, None));
    }

    #[test]
    #[rustfmt::skip]
    fn test_trim_multiline_empty_line() {
        assert_eq!("\
    if x {
        y

    } else {
        z
    }", trim_multiline("    if x {
            y

        } else {
            z
        }".into(), false, None));
    }

    #[test]
    fn test_without_block_comments_lines_without_block_comments() {
        let result = without_block_comments(vec!["/*", "", "*/"]);
        println!("result: {:?}", result);
        assert!(result.is_empty());

        let result = without_block_comments(vec!["", "/*", "", "*/", "#[crate_type = \"lib\"]", "/*", "", "*/", ""]);
        assert_eq!(result, vec!["", "#[crate_type = \"lib\"]", ""]);

        let result = without_block_comments(vec!["/* rust", "", "*/"]);
        assert!(result.is_empty());

        let result = without_block_comments(vec!["/* one-line comment */"]);
        assert!(result.is_empty());

        let result = without_block_comments(vec!["/* nested", "/* multi-line", "comment", "*/", "test", "*/"]);
        assert!(result.is_empty());

        let result = without_block_comments(vec!["/* nested /* inline /* comment */ test */ */"]);
        assert!(result.is_empty());

        let result = without_block_comments(vec!["foo", "bar", "baz"]);
        assert_eq!(result, vec!["foo", "bar", "baz"]);
    }
}
