//! checks for attributes

use crate::reexport::*;
use crate::utils::{
    in_macro_or_desugar, is_present_in_source, last_line_of_span, match_def_path, paths, snippet_opt, span_lint,
    span_lint_and_sugg, span_lint_and_then, without_block_comments,
};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{
    in_external_macro, CheckLintNameResult, EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintArray,
    LintContext, LintPass,
};
use rustc::ty;
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use semver::Version;
use syntax::ast::{AttrStyle, Attribute, Lit, LitKind, MetaItemKind, NestedMetaItem};
use syntax::source_map::Span;
use syntax_pos::symbol::Symbol;

declare_clippy_lint! {
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
    /// ```ignore
    /// #[inline(always)]
    /// fn not_quite_hot_code(..) { ... }
    /// ```
    pub INLINE_ALWAYS,
    pedantic,
    "use of `#[inline(always)]`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `extern crate` and `use` items annotated with
    /// lint attributes.
    ///
    /// This lint whitelists `#[allow(unused_imports)]`, `#[allow(deprecated)]` and
    /// `#[allow(unreachable_pub)]` on `use` items and `#[allow(unused_imports)]` on
    /// `extern crate` items with a `#[macro_use]` attribute.
    ///
    /// **Why is this bad?** Lint attributes have no effect on crate imports. Most
    /// likely a `!` was forgotten.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// // Bad
    /// #[deny(dead_code)]
    /// extern crate foo;
    /// #[forbid(dead_code)]
    /// use foo::bar;
    ///
    /// // Ok
    /// #[allow(unused_imports)]
    /// use foo::baz;
    /// #[allow(unused_imports)]
    /// #[macro_use]
    /// extern crate baz;
    /// ```
    pub USELESS_ATTRIBUTE,
    correctness,
    "use of lint attributes on `extern crate` items"
}

declare_clippy_lint! {
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
    /// fn something_else() { /* ... */ }
    /// ```
    pub DEPRECATED_SEMVER,
    correctness,
    "use of `#[deprecated(since = \"x\")]` where x is not semver"
}

declare_clippy_lint! {
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
    /// // Good (as inner attribute)
    /// #![inline(always)]
    ///
    /// fn this_is_fine() { }
    ///
    /// // Bad
    /// #[inline(always)]
    ///
    /// fn not_quite_good_code() { }
    ///
    /// // Good (as outer attribute)
    /// #[inline(always)]
    /// fn this_is_fine_too() { }
    /// ```
    pub EMPTY_LINE_AFTER_OUTER_ATTR,
    nursery,
    "empty line after outer attribute"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `allow`/`warn`/`deny`/`forbid` attributes with scoped clippy
    /// lints and if those lints exist in clippy. If there is an uppercase letter in the lint name
    /// (not the tool name) and a lowercase version of this lint exists, it will suggest to lowercase
    /// the lint name.
    ///
    /// **Why is this bad?** A lint attribute with a mistyped lint name won't have an effect.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust
    /// #![warn(if_not_els)]
    /// #![deny(clippy::All)]
    /// ```
    ///
    /// Good:
    /// ```rust
    /// #![warn(if_not_else)]
    /// #![deny(clippy::all)]
    /// ```
    pub UNKNOWN_CLIPPY_LINTS,
    style,
    "unknown_lints for scoped Clippy lints"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `#[cfg_attr(rustfmt, rustfmt_skip)]` and suggests to replace it
    /// with `#[rustfmt::skip]`.
    ///
    /// **Why is this bad?** Since tool_attributes ([rust-lang/rust#44690](https://github.com/rust-lang/rust/issues/44690))
    /// are stable now, they should be used instead of the old `cfg_attr(rustfmt)` attributes.
    ///
    /// **Known problems:** This lint doesn't detect crate level inner attributes, because they get
    /// processed before the PreExpansionPass lints get executed. See
    /// [#3123](https://github.com/rust-lang/rust-clippy/pull/3123#issuecomment-422321765)
    ///
    /// **Example:**
    ///
    /// Bad:
    /// ```rust
    /// #[cfg_attr(rustfmt, rustfmt_skip)]
    /// fn main() { }
    /// ```
    ///
    /// Good:
    /// ```rust
    /// #[rustfmt::skip]
    /// fn main() { }
    /// ```
    pub DEPRECATED_CFG_ATTR,
    complexity,
    "usage of `cfg_attr(rustfmt)` instead of `tool_attributes`"
}

declare_lint_pass!(Attributes => [
    INLINE_ALWAYS,
    DEPRECATED_SEMVER,
    USELESS_ATTRIBUTE,
    EMPTY_LINE_AFTER_OUTER_ATTR,
    UNKNOWN_CLIPPY_LINTS,
]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Attributes {
    fn check_attribute(&mut self, cx: &LateContext<'a, 'tcx>, attr: &'tcx Attribute) {
        if let Some(items) = &attr.meta_item_list() {
            if let Some(ident) = attr.ident() {
                match &*ident.as_str() {
                    "allow" | "warn" | "deny" | "forbid" => {
                        check_clippy_lint_names(cx, items);
                    },
                    _ => {},
                }
                if items.is_empty() || !attr.check_name(sym!(deprecated)) {
                    return;
                }
                for item in items {
                    if_chain! {
                        if let NestedMetaItem::MetaItem(mi) = &item;
                        if let MetaItemKind::NameValue(lit) = &mi.node;
                        if mi.check_name(sym!(since));
                        then {
                            check_semver(cx, item.span(), lit);
                        }
                    }
                }
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if is_relevant_item(cx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
        match item.node {
            ItemKind::ExternCrate(..) | ItemKind::Use(..) => {
                let skip_unused_imports = item.attrs.iter().any(|attr| attr.check_name(sym!(macro_use)));

                for attr in &item.attrs {
                    if in_external_macro(cx.sess(), attr.span) {
                        return;
                    }
                    if let Some(lint_list) = &attr.meta_item_list() {
                        if let Some(ident) = attr.ident() {
                            match &*ident.as_str() {
                                "allow" | "warn" | "deny" | "forbid" => {
                                    // whitelist `unused_imports`, `deprecated` and `unreachable_pub` for `use` items
                                    // and `unused_imports` for `extern crate` items with `macro_use`
                                    for lint in lint_list {
                                        match item.node {
                                            ItemKind::Use(..) => {
                                                if is_word(lint, sym!(unused_imports))
                                                    || is_word(lint, sym!(deprecated))
                                                    || is_word(lint, sym!(unreachable_pub))
                                                {
                                                    return;
                                                }
                                            },
                                            ItemKind::ExternCrate(..) => {
                                                if is_word(lint, sym!(unused_imports)) && skip_unused_imports {
                                                    return;
                                                }
                                                if is_word(lint, sym!(unused_extern_crates)) {
                                                    return;
                                                }
                                            },
                                            _ => {},
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
                                                    db.span_suggestion(
                                                        line_span,
                                                        "if you just forgot a `!`, use",
                                                        sugg,
                                                        Applicability::MachineApplicable,
                                                    );
                                                },
                                            );
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
        if is_relevant_impl(cx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        if is_relevant_trait(cx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
    }
}

#[allow(clippy::single_match_else)]
fn check_clippy_lint_names(cx: &LateContext<'_, '_>, items: &[NestedMetaItem]) {
    let lint_store = cx.lints();
    for lint in items {
        if_chain! {
            if let Some(meta_item) = lint.meta_item();
            if meta_item.path.segments.len() > 1;
            if let tool_name = meta_item.path.segments[0].ident;
            if tool_name.as_str() == "clippy";
            let name = meta_item.path.segments.last().unwrap().ident.name;
            if let CheckLintNameResult::Tool(Err((None, _))) = lint_store.check_lint_name(
                &name.as_str(),
                Some(tool_name.as_str()),
            );
            then {
                span_lint_and_then(
                    cx,
                    UNKNOWN_CLIPPY_LINTS,
                    lint.span(),
                    &format!("unknown clippy lint: clippy::{}", name),
                    |db| {
                        if name.as_str().chars().any(char::is_uppercase) {
                            let name_lower = name.as_str().to_lowercase();
                            match lint_store.check_lint_name(
                                &name_lower,
                                Some(tool_name.as_str())
                            ) {
                                // FIXME: can we suggest similar lint names here?
                                // https://github.com/rust-lang/rust/pull/56992
                                CheckLintNameResult::NoLint(None) => (),
                                _ => {
                                    db.span_suggestion(
                                        lint.span(),
                                        "lowercase the lint name",
                                        name_lower,
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            }
                        }
                    }
                );
            }
        };
    }
}

fn is_relevant_item(cx: &LateContext<'_, '_>, item: &Item) -> bool {
    if let ItemKind::Fn(_, _, _, eid) = item.node {
        is_relevant_expr(cx, cx.tcx.body_tables(eid), &cx.tcx.hir().body(eid).value)
    } else {
        true
    }
}

fn is_relevant_impl(cx: &LateContext<'_, '_>, item: &ImplItem) -> bool {
    match item.node {
        ImplItemKind::Method(_, eid) => is_relevant_expr(cx, cx.tcx.body_tables(eid), &cx.tcx.hir().body(eid).value),
        _ => false,
    }
}

fn is_relevant_trait(cx: &LateContext<'_, '_>, item: &TraitItem) -> bool {
    match item.node {
        TraitItemKind::Method(_, TraitMethod::Required(_)) => true,
        TraitItemKind::Method(_, TraitMethod::Provided(eid)) => {
            is_relevant_expr(cx, cx.tcx.body_tables(eid), &cx.tcx.hir().body(eid).value)
        },
        _ => false,
    }
}

fn is_relevant_block(cx: &LateContext<'_, '_>, tables: &ty::TypeckTables<'_>, block: &Block) -> bool {
    if let Some(stmt) = block.stmts.first() {
        match &stmt.node {
            StmtKind::Local(_) => true,
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => is_relevant_expr(cx, tables, expr),
            _ => false,
        }
    } else {
        block.expr.as_ref().map_or(false, |e| is_relevant_expr(cx, tables, e))
    }
}

fn is_relevant_expr(cx: &LateContext<'_, '_>, tables: &ty::TypeckTables<'_>, expr: &Expr) -> bool {
    match &expr.node {
        ExprKind::Block(block, _) => is_relevant_block(cx, tables, block),
        ExprKind::Ret(Some(e)) => is_relevant_expr(cx, tables, e),
        ExprKind::Ret(None) | ExprKind::Break(_, None) => false,
        ExprKind::Call(path_expr, _) => {
            if let ExprKind::Path(qpath) = &path_expr.node {
                if let Some(fun_id) = tables.qpath_res(qpath, path_expr.hir_id).opt_def_id() {
                    !match_def_path(cx, fun_id, &paths::BEGIN_PANIC)
                } else {
                    true
                }
            } else {
                true
            }
        },
        _ => true,
    }
}

fn check_attrs(cx: &LateContext<'_, '_>, span: Span, name: Name, attrs: &[Attribute]) {
    if in_macro_or_desugar(span) {
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
                        "Found an empty line after an outer attribute. \
                         Perhaps you forgot to add a '!' to make it an inner attribute?",
                    );
                }
            }
        }

        if let Some(values) = attr.meta_item_list() {
            if values.len() != 1 || !attr.check_name(sym!(inline)) {
                continue;
            }
            if is_word(&values[0], sym!(always)) {
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

fn check_semver(cx: &LateContext<'_, '_>, span: Span, lit: &Lit) {
    if let LitKind::Str(is, _) = lit.node {
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

fn is_word(nmi: &NestedMetaItem, expected: Symbol) -> bool {
    if let NestedMetaItem::MetaItem(mi) = &nmi {
        mi.is_word() && mi.check_name(expected)
    } else {
        false
    }
}

declare_lint_pass!(DeprecatedCfgAttribute => [DEPRECATED_CFG_ATTR]);

impl EarlyLintPass for DeprecatedCfgAttribute {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &Attribute) {
        if_chain! {
            // check cfg_attr
            if attr.check_name(sym!(cfg_attr));
            if let Some(items) = attr.meta_item_list();
            if items.len() == 2;
            // check for `rustfmt`
            if let Some(feature_item) = items[0].meta_item();
            if feature_item.check_name(sym!(rustfmt));
            // check for `rustfmt_skip` and `rustfmt::skip`
            if let Some(skip_item) = &items[1].meta_item();
            if skip_item.check_name(sym!(rustfmt_skip)) ||
                skip_item.path.segments.last().expect("empty path in attribute").ident.name == sym!(skip);
            // Only lint outer attributes, because custom inner attributes are unstable
            // Tracking issue: https://github.com/rust-lang/rust/issues/54726
            if let AttrStyle::Outer = attr.style;
            then {
                span_lint_and_sugg(
                    cx,
                    DEPRECATED_CFG_ATTR,
                    attr.span,
                    "`cfg_attr` is deprecated for rustfmt and got replaced by tool_attributes",
                    "use",
                    "#[rustfmt::skip]".to_string(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
