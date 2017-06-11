//! checks for attributes

use reexport::*;
use rustc::lint::*;
use rustc::hir::*;
use rustc::ty::{self, TyCtxt};
use semver::Version;
use syntax::ast::{Attribute, Lit, LitKind, MetaItemKind, NestedMetaItem, NestedMetaItemKind};
use syntax::codemap::Span;
use utils::{in_macro, match_def_path, paths, span_lint, span_lint_and_then, snippet_opt};

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
declare_lint! {
    pub INLINE_ALWAYS,
    Warn,
    "use of `#[inline(always)]`"
}

/// **What it does:** Checks for `extern crate` and `use` items annotated with lint attributes
///
/// **Why is this bad?** Lint attributes have no effect on crate imports. Most likely a `!` was
/// forgotten
///
/// **Known problems:** Technically one might allow `unused_import` on a `use` item,
/// but it's easier to remove the unused item.
///
/// **Example:**
/// ```rust
/// #[deny(dead_code)]
/// extern crate foo;
/// #[allow(unused_import)]
/// use foo::bar;
/// ```
declare_lint! {
    pub USELESS_ATTRIBUTE,
    Warn,
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
declare_lint! {
    pub DEPRECATED_SEMVER,
    Warn,
    "use of `#[deprecated(since = \"x\")]` where x is not semver"
}

#[derive(Copy,Clone)]
pub struct AttrPass;

impl LintPass for AttrPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INLINE_ALWAYS, DEPRECATED_SEMVER, USELESS_ATTRIBUTE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AttrPass {
    fn check_attribute(&mut self, cx: &LateContext<'a, 'tcx>, attr: &'tcx Attribute) {
        if let Some(ref items) = attr.meta_item_list() {
            if items.is_empty() || attr.name().map_or(true, |n| n != "deprecated") {
                return;
            }
            for item in items {
                if_let_chain! {[
                    let NestedMetaItemKind::MetaItem(ref mi) = item.node,
                    let MetaItemKind::NameValue(ref lit) = mi.node,
                    mi.name() == "since",
                ], {
                    check_semver(cx, item.span, lit);
                }}
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if is_relevant_item(cx.tcx, item) {
            check_attrs(cx, item.span, &item.name, &item.attrs)
        }
        match item.node {
            ItemExternCrate(_) |
            ItemUse(_, _) => {
                for attr in &item.attrs {
                    if let Some(ref lint_list) = attr.meta_item_list() {
                        if let Some(name) = attr.name() {
                            match &*name.as_str() {
                                "allow" | "warn" | "deny" | "forbid" => {
                                    // whitelist `unused_imports` and `deprecated`
                                    for lint in lint_list {
                                        if is_word(lint, "unused_imports") || is_word(lint, "deprecated") {
                                            if let ItemUse(_, _) = item.node {
                                                return;
                                            }
                                        }
                                    }
                                    if let Some(mut sugg) = snippet_opt(cx, attr.span) {
                                        if sugg.len() > 1 {
                                            span_lint_and_then(cx,
                                                               USELESS_ATTRIBUTE,
                                                               attr.span,
                                                               "useless lint attribute",
                                                               |db| {
                                                sugg.insert(1, '!');
                                                db.span_suggestion(attr.span, "if you just forgot a `!`, use", sugg);
                                            });
                                        }
                                    }
                                },
                                _ => {},
                            }
                        }
                    }
                }
            },
            _ => {},
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx ImplItem) {
        if is_relevant_impl(cx.tcx, item) {
            check_attrs(cx, item.span, &item.name, &item.attrs)
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        if is_relevant_trait(cx.tcx, item) {
            check_attrs(cx, item.span, &item.name, &item.attrs)
        }
    }
}

fn is_relevant_item(tcx: TyCtxt, item: &Item) -> bool {
    if let ItemFn(_, _, _, _, _, eid) = item.node {
        is_relevant_expr(tcx, tcx.body_tables(eid), &tcx.hir.body(eid).value)
    } else {
        false
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
    for stmt in &block.stmts {
        match stmt.node {
            StmtDecl(_, _) => return true,
            StmtExpr(ref expr, _) |
            StmtSemi(ref expr, _) => {
                return is_relevant_expr(tcx, tables, expr);
            },
        }
    }
    block.expr.as_ref().map_or(false, |e| is_relevant_expr(tcx, tables, e))
}

fn is_relevant_expr(tcx: TyCtxt, tables: &ty::TypeckTables, expr: &Expr) -> bool {
    match expr.node {
        ExprBlock(ref block) => is_relevant_block(tcx, tables, block),
        ExprRet(Some(ref e)) => is_relevant_expr(tcx, tables, e),
        ExprRet(None) |
        ExprBreak(_, None) => false,
        ExprCall(ref path_expr, _) => {
            if let ExprPath(ref qpath) = path_expr.node {
                let fun_id = tables.qpath_def(qpath, path_expr.id).def_id();
                !match_def_path(tcx, fun_id, &paths::BEGIN_PANIC)
            } else {
                true
            }
        },
        _ => true,
    }
}

fn check_attrs(cx: &LateContext, span: Span, name: &Name, attrs: &[Attribute]) {
    if in_macro(span) {
        return;
    }

    for attr in attrs {
        if let Some(ref values) = attr.meta_item_list() {
            if values.len() != 1 || attr.name().map_or(true, |n| n != "inline") {
                continue;
            }
            if is_word(&values[0], "always") {
                span_lint(cx,
                          INLINE_ALWAYS,
                          attr.span,
                          &format!("you have declared `#[inline(always)]` on `{}`. This is usually a bad idea",
                                   name));
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
    span_lint(cx,
              DEPRECATED_SEMVER,
              span,
              "the since field must contain a semver-compliant version");
}

fn is_word(nmi: &NestedMetaItem, expected: &str) -> bool {
    if let NestedMetaItemKind::MetaItem(ref mi) = nmi.node {
        mi.is_word() && mi.name() == expected
    } else {
        false
    }
}
