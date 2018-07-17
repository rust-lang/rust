//! checks for attributes

use crate::reexport::*;
use crate::utils::{
    in_macro, last_line_of_span, match_def_path, opt_def_id, paths, snippet_opt, span_lint, span_lint_and_then,
    without_block_comments,
};
use rustc::hir::*;
use rustc::lint::*;
use rustc::ty::{self, TyCtxt};
use semver::Version;
use syntax::ast::{AttrStyle, Attribute, Lit, LitKind, MetaItemKind, NestedMetaItem, NestedMetaItemKind};
use syntax::codemap::Span;

/// **What it does:** Checks for items annotated with `#[inline(always)]`,
/// unless the annotated function is empty or simply panics.
///
/// **Why is this bad?** While there are valid uses of this annotation (and once
/// you know when to use it, by all means `allow` this lint), it's a common
/// newbie-mistake to pepper one's code with it.
///
/// As a rule of thumb, before slapping `#[inline(always)]` on a function,
/// measure if that additional function call really affects your runtime profile
/// sufficiently to make up for the increase in compile time.
///
/// **Known problems:** False positives, big time. This lint is meant to be
/// deactivated by everyone doing serious performance work. This means having
/// done the measurement.
///
/// **Example:**
/// ```rust
/// #[inline(always)]
/// fn not_quite_hot_code(..) { ... }
/// ```
declare_clippy_lint! {
    pub INLINE_ALWAYS,
    pedantic,
    "use of `#[inline(always)]`"
}

/// **What it does:** Checks for `extern crate` and `use` items annotated with
/// lint attributes
///
/// **Why is this bad?** Lint attributes have no effect on crate imports. Most
/// likely a `!` was
/// forgotten
///
/// **Known problems:** Technically one might allow `unused_import` on a `use`
/// item,
/// but it's easier to remove the unused item.
///
/// **Example:**
/// ```rust
/// #[deny(dead_code)]
/// extern crate foo;
/// #[allow(unused_import)]
/// use foo::bar;
/// ```
declare_clippy_lint! {
    pub USELESS_ATTRIBUTE,
    correctness,
    "use of lint attributes on `extern crate` items"
}

/// **What it does:** Checks for `#[deprecated]` annotations with a `since`
/// field that is not a valid semantic version.
///
/// **Why is this bad?** For checking the version of the deprecation, it must be
/// a valid semver. Failing that, the contained information is useless.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// #[deprecated(since = "forever")]
/// fn something_else(..) { ... }
/// ```
declare_clippy_lint! {
    pub DEPRECATED_SEMVER,
    correctness,
    "use of `#[deprecated(since = \"x\")]` where x is not semver"
}

/// **What it does:** Checks for empty lines after outer attributes
///
/// **Why is this bad?**
/// Most likely the attribute was meant to be an inner attribute using a '!'.
/// If it was meant to be an outer attribute, then the following item
/// should not be separated by empty lines.
///
/// **Known problems:** Can cause false positives.
///
/// From the clippy side it's difficult to detect empty lines between an attributes and the
/// following item because empty lines and comments are not part of the AST. The parsing
/// currently works for basic cases but is not perfect.
///
/// **Example:**
/// ```rust
/// // Bad
/// #[inline(always)]
///
/// fn not_quite_good_code(..) { ... }
///
/// // Good (as inner attribute)
/// #![inline(always)]
///
/// fn this_is_fine(..) { ... }
///
/// // Good (as outer attribute)
/// #[inline(always)]
/// fn this_is_fine_too(..) { ... }
/// ```
declare_clippy_lint! {
    pub EMPTY_LINE_AFTER_OUTER_ATTR,
    nursery,
    "empty line after outer attribute"
}

#[derive(Copy, Clone)]
pub struct AttrPass;

impl LintPass for AttrPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            INLINE_ALWAYS,
            DEPRECATED_SEMVER,
            USELESS_ATTRIBUTE,
            EMPTY_LINE_AFTER_OUTER_ATTR
        )
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AttrPass {
    fn check_attribute(&mut self, cx: &LateContext<'a, 'tcx>, attr: &'tcx Attribute) {
        if let Some(ref items) = attr.meta_item_list() {
            if items.is_empty() || attr.name() != "deprecated" {
                return;
            }
            for item in items {
                if_chain! {
                    if let NestedMetaItemKind::MetaItem(ref mi) = item.node;
                    if let MetaItemKind::NameValue(ref lit) = mi.node;
                    if mi.name() == "since";
                    then {
                        check_semver(cx, item.span, lit);
                    }
                }
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if is_relevant_item(cx.tcx, item) {
            check_attrs(cx, item.span, item.name, &item.attrs)
        }
        match item.node {
            ItemKind::ExternCrate(_) | ItemKind::Use(_, _) => {
                for attr in &item.attrs {
                    if let Some(ref lint_list) = attr.meta_item_list() {
                        match &*attr.name().as_str() {
                            "allow" | "warn" | "deny" | "forbid" => {
                                // whitelist `unused_imports` and `deprecated`
                                for lint in lint_list {
                                    if is_word(lint, "unused_imports") || is_word(lint, "deprecated") {
                                        if let ItemKind::Use(_, _) = item.node {
                                            return;
                                        }
                                    }
                                }
                                let line_span = last_line_of_span(cx, attr.span);

                                if let Some(mut sugg) = snippet_opt(cx, line_span) {
                                    if sugg.contains("#[") {
                                        span_lint_and_then(
                                            cx,
                                            USELESS_ATTRIBUTE,
                                            line_span,
                                            "useless lint attribute",
                                            |db| {
                                                sugg = sugg.replacen("#[", "#![", 1);
                                                db.span_suggestion(line_span, "if you just forgot a `!`, use", sugg);
                                            },
                                        );
                                    }
                                }
                            },
                            _ => {},
                        }
                    }
                }
            },
            _ => {},
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx ImplItem) {
        if is_relevant_impl(cx.tcx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        if is_relevant_trait(cx.tcx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
    }
}

fn is_relevant_item(tcx: TyCtxt, item: &Item) -> bool {
    if let ItemKind::Fn(_, _, _, eid) = item.node {
        is_relevant_expr(tcx, tcx.body_tables(eid), &tcx.hir.body(eid).value)
    } else {
        true
    }
}

fn is_relevant_impl(tcx: TyCtxt, item: &ImplItem) -> bool {
    match item.node {
        ImplItemKind::Method(_, eid) => is_relevant_expr(tcx, tcx.body_tables(eid), &tcx.hir.body(eid).value),
        _ => false,
    }
}

fn is_relevant_trait(tcx: TyCtxt, item: &TraitItem) -> bool {
    match item.node {
        TraitItemKind::Method(_, TraitMethod::Required(_)) => true,
        TraitItemKind::Method(_, TraitMethod::Provided(eid)) => {
            is_relevant_expr(tcx, tcx.body_tables(eid), &tcx.hir.body(eid).value)
        },
        _ => false,
    }
}

fn is_relevant_block(tcx: TyCtxt, tables: &ty::TypeckTables, block: &Block) -> bool {
    if let Some(stmt) = block.stmts.first() {
        match stmt.node {
            StmtKind::Decl(_, _) => true,
            StmtKind::Expr(ref expr, _) | StmtKind::Semi(ref expr, _) => is_relevant_expr(tcx, tables, expr),
        }
    } else {
        block.expr.as_ref().map_or(false, |e| is_relevant_expr(tcx, tables, e))
    }
}

fn is_relevant_expr(tcx: TyCtxt, tables: &ty::TypeckTables, expr: &Expr) -> bool {
    match expr.node {
        ExprKind::Block(ref block, _) => is_relevant_block(tcx, tables, block),
        ExprKind::Ret(Some(ref e)) => is_relevant_expr(tcx, tables, e),
        ExprKind::Ret(None) | ExprKind::Break(_, None) => false,
        ExprKind::Call(ref path_expr, _) => if let ExprKind::Path(ref qpath) = path_expr.node {
            if let Some(fun_id) = opt_def_id(tables.qpath_def(qpath, path_expr.hir_id)) {
                !match_def_path(tcx, fun_id, &paths::BEGIN_PANIC)
            } else {
                true
            }
        } else {
            true
        },
        _ => true,
    }
}

fn check_attrs(cx: &LateContext, span: Span, name: Name, attrs: &[Attribute]) {
    if in_macro(span) {
        return;
    }

    for attr in attrs {
        if attr.is_sugared_doc {
            return;
        }
        if attr.style == AttrStyle::Outer {
            if attr.tokens.is_empty() || !is_present_in_source(cx, attr.span) {
                return;
            }

            let begin_of_attr_to_item = Span::new(attr.span.lo(), span.lo(), span.ctxt());
            let end_of_attr_to_item = Span::new(attr.span.hi(), span.lo(), span.ctxt());

            if let Some(snippet) = snippet_opt(cx, end_of_attr_to_item) {
                let lines = snippet.split('\n').collect::<Vec<_>>();
                let lines = without_block_comments(lines);

                if lines.iter().filter(|l| l.trim().is_empty()).count() > 2 {
                    span_lint(
                        cx,
                        EMPTY_LINE_AFTER_OUTER_ATTR,
                        begin_of_attr_to_item,
                        "Found an empty line after an outer attribute. Perhaps you forgot to add a '!' to make it an inner attribute?"
                    );
                }
            }
        }

        if let Some(ref values) = attr.meta_item_list() {
            if values.len() != 1 || attr.name() != "inline" {
                continue;
            }
            if is_word(&values[0], "always") {
                span_lint(
                    cx,
                    INLINE_ALWAYS,
                    attr.span,
                    &format!(
                        "you have declared `#[inline(always)]` on `{}`. This is usually a bad idea",
                        name
                    ),
                );
            }
        }
    }
}

fn check_semver(cx: &LateContext, span: Span, lit: &Lit) {
    if let LitKind::Str(ref is, _) = lit.node {
        if Version::parse(&is.as_str()).is_ok() {
            return;
        }
    }
    span_lint(
        cx,
        DEPRECATED_SEMVER,
        span,
        "the since field must contain a semver-compliant version",
    );
}

fn is_word(nmi: &NestedMetaItem, expected: &str) -> bool {
    if let NestedMetaItemKind::MetaItem(ref mi) = nmi.node {
        mi.is_word() && mi.name() == expected
    } else {
        false
    }
}

// If the snippet is empty, it's an attribute that was inserted during macro
// expansion and we want to ignore those, because they could come from external
// sources that the user has no control over.
// For some reason these attributes don't have any expansion info on them, so
// we have to check it this way until there is a better way.
fn is_present_in_source(cx: &LateContext, span: Span) -> bool {
    if let Some(snippet) = snippet_opt(cx, span) {
        if snippet.is_empty() {
            return false;
        }
    }
    true
}
