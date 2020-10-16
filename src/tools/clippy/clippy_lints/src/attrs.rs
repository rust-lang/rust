//! checks for attributes

use crate::utils::{
    first_line_of_span, is_present_in_source, match_def_path, paths, snippet_opt, span_lint, span_lint_and_help,
    span_lint_and_sugg, span_lint_and_then, without_block_comments,
};
use if_chain::if_chain;
use rustc_ast::util::lev_distance::find_best_match_for_name;
use rustc_ast::{AttrKind, AttrStyle, Attribute, Lit, LitKind, MetaItemKind, NestedMetaItem};
use rustc_errors::Applicability;
use rustc_hir::{
    Block, Expr, ExprKind, ImplItem, ImplItemKind, Item, ItemKind, StmtKind, TraitFn, TraitItem, TraitItemKind,
};
use rustc_lint::{CheckLintNameResult, EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::symbol::{Symbol, SymbolStr};
use semver::Version;

static UNIX_SYSTEMS: &[&str] = &[
    "android",
    "dragonfly",
    "emscripten",
    "freebsd",
    "fuchsia",
    "haiku",
    "illumos",
    "ios",
    "l4re",
    "linux",
    "macos",
    "netbsd",
    "openbsd",
    "redox",
    "solaris",
    "vxworks",
];

// NOTE: windows is excluded from the list because it's also a valid target family.
static NON_UNIX_SYSTEMS: &[&str] = &["cloudabi", "hermit", "none", "wasi"];

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
    /// This lint permits `#[allow(unused_imports)]`, `#[allow(deprecated)]`,
    /// `#[allow(unreachable_pub)]`, `#[allow(clippy::wildcard_imports)]` and
    /// `#[allow(clippy::enum_glob_use)]` on `use` items and `#[allow(unused_imports)]` on
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
    /// #![allow(dead_code)]
    ///
    /// fn this_is_fine() { }
    ///
    /// // Bad
    /// #[allow(dead_code)]
    ///
    /// fn not_quite_good_code() { }
    ///
    /// // Good (as outer attribute)
    /// #[allow(dead_code)]
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
    /// **What it does:** Checks for `warn`/`deny`/`forbid` attributes targeting the whole clippy::restriction category.
    ///
    /// **Why is this bad?** Restriction lints sometimes are in contrast with other lints or even go against idiomatic rust.
    /// These lints should only be enabled on a lint-by-lint basis and with careful consideration.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust
    /// #![deny(clippy::restriction)]
    /// ```
    ///
    /// Good:
    /// ```rust
    /// #![deny(clippy::as_conversions)]
    /// ```
    pub BLANKET_CLIPPY_RESTRICTION_LINTS,
    style,
    "enabling the complete restriction group"
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
    "usage of `cfg_attr(rustfmt)` instead of tool attributes"
}

declare_clippy_lint! {
    /// **What it does:** Checks for cfg attributes having operating systems used in target family position.
    ///
    /// **Why is this bad?** The configuration option will not be recognised and the related item will not be included
    /// by the conditional compilation engine.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// Bad:
    /// ```rust
    /// #[cfg(linux)]
    /// fn conditional() { }
    /// ```
    ///
    /// Good:
    /// ```rust
    /// #[cfg(target_os = "linux")]
    /// fn conditional() { }
    /// ```
    ///
    /// Or:
    /// ```rust
    /// #[cfg(unix)]
    /// fn conditional() { }
    /// ```
    /// Check the [Rust Reference](https://doc.rust-lang.org/reference/conditional-compilation.html#target_os) for more details.
    pub MISMATCHED_TARGET_OS,
    correctness,
    "usage of `cfg(operating_system)` instead of `cfg(target_os = \"operating_system\")`"
}

declare_lint_pass!(Attributes => [
    INLINE_ALWAYS,
    DEPRECATED_SEMVER,
    USELESS_ATTRIBUTE,
    UNKNOWN_CLIPPY_LINTS,
    BLANKET_CLIPPY_RESTRICTION_LINTS,
]);

impl<'tcx> LateLintPass<'tcx> for Attributes {
    fn check_attribute(&mut self, cx: &LateContext<'tcx>, attr: &'tcx Attribute) {
        if let Some(items) = &attr.meta_item_list() {
            if let Some(ident) = attr.ident() {
                let ident = &*ident.as_str();
                match ident {
                    "allow" | "warn" | "deny" | "forbid" => {
                        check_clippy_lint_names(cx, ident, items);
                    },
                    _ => {},
                }
                if items.is_empty() || !attr.has_name(sym!(deprecated)) {
                    return;
                }
                for item in items {
                    if_chain! {
                        if let NestedMetaItem::MetaItem(mi) = &item;
                        if let MetaItemKind::NameValue(lit) = &mi.kind;
                        if mi.has_name(sym!(since));
                        then {
                            check_semver(cx, item.span(), lit);
                        }
                    }
                }
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if is_relevant_item(cx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
        match item.kind {
            ItemKind::ExternCrate(..) | ItemKind::Use(..) => {
                let skip_unused_imports = item.attrs.iter().any(|attr| attr.has_name(sym!(macro_use)));

                for attr in item.attrs {
                    if in_external_macro(cx.sess(), attr.span) {
                        return;
                    }
                    if let Some(lint_list) = &attr.meta_item_list() {
                        if let Some(ident) = attr.ident() {
                            match &*ident.as_str() {
                                "allow" | "warn" | "deny" | "forbid" => {
                                    // permit `unused_imports`, `deprecated`, `unreachable_pub`,
                                    // `clippy::wildcard_imports`, and `clippy::enum_glob_use` for `use` items
                                    // and `unused_imports` for `extern crate` items with `macro_use`
                                    for lint in lint_list {
                                        match item.kind {
                                            ItemKind::Use(..) => {
                                                if is_word(lint, sym!(unused_imports))
                                                    || is_word(lint, sym!(deprecated))
                                                    || is_word(lint, sym!(unreachable_pub))
                                                    || is_word(lint, sym!(unused))
                                                    || extract_clippy_lint(lint)
                                                        .map_or(false, |s| s == "wildcard_imports")
                                                    || extract_clippy_lint(lint).map_or(false, |s| s == "enum_glob_use")
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
                                    let line_span = first_line_of_span(cx, attr.span);

                                    if let Some(mut sugg) = snippet_opt(cx, line_span) {
                                        if sugg.contains("#[") {
                                            span_lint_and_then(
                                                cx,
                                                USELESS_ATTRIBUTE,
                                                line_span,
                                                "useless lint attribute",
                                                |diag| {
                                                    sugg = sugg.replacen("#[", "#![", 1);
                                                    diag.span_suggestion(
                                                        line_span,
                                                        "if you just forgot a `!`, use",
                                                        sugg,
                                                        Applicability::MaybeIncorrect,
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

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if is_relevant_impl(cx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if is_relevant_trait(cx, item) {
            check_attrs(cx, item.span, item.ident.name, &item.attrs)
        }
    }
}

/// Returns the lint name if it is clippy lint.
fn extract_clippy_lint(lint: &NestedMetaItem) -> Option<SymbolStr> {
    if_chain! {
        if let Some(meta_item) = lint.meta_item();
        if meta_item.path.segments.len() > 1;
        if let tool_name = meta_item.path.segments[0].ident;
        if tool_name.as_str() == "clippy";
        let lint_name = meta_item.path.segments.last().unwrap().ident.name;
        then {
            return Some(lint_name.as_str());
        }
    }
    None
}

fn check_clippy_lint_names(cx: &LateContext<'_>, ident: &str, items: &[NestedMetaItem]) {
    let lint_store = cx.lints();
    for lint in items {
        if let Some(lint_name) = extract_clippy_lint(lint) {
            if let CheckLintNameResult::Tool(Err((None, _))) =
                lint_store.check_lint_name(&lint_name, Some(sym!(clippy)))
            {
                span_lint_and_then(
                    cx,
                    UNKNOWN_CLIPPY_LINTS,
                    lint.span(),
                    &format!("unknown clippy lint: clippy::{}", lint_name),
                    |diag| {
                        let name_lower = lint_name.to_lowercase();
                        let symbols = lint_store
                            .get_lints()
                            .iter()
                            .map(|l| Symbol::intern(&l.name_lower()))
                            .collect::<Vec<_>>();
                        let sugg = find_best_match_for_name(
                            symbols.iter(),
                            Symbol::intern(&format!("clippy::{}", name_lower)),
                            None,
                        );
                        if lint_name.chars().any(char::is_uppercase)
                            && lint_store.find_lints(&format!("clippy::{}", name_lower)).is_ok()
                        {
                            diag.span_suggestion(
                                lint.span(),
                                "lowercase the lint name",
                                format!("clippy::{}", name_lower),
                                Applicability::MachineApplicable,
                            );
                        } else if let Some(sugg) = sugg {
                            diag.span_suggestion(
                                lint.span(),
                                "did you mean",
                                sugg.to_string(),
                                Applicability::MachineApplicable,
                            );
                        }
                    },
                );
            } else if lint_name == "restriction" && ident != "allow" {
                span_lint_and_help(
                    cx,
                    BLANKET_CLIPPY_RESTRICTION_LINTS,
                    lint.span(),
                    "restriction lints are not meant to be all enabled",
                    None,
                    "try enabling only the lints you really need",
                );
            }
        }
    }
}

fn is_relevant_item(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if let ItemKind::Fn(_, _, eid) = item.kind {
        is_relevant_expr(cx, cx.tcx.typeck_body(eid), &cx.tcx.hir().body(eid).value)
    } else {
        true
    }
}

fn is_relevant_impl(cx: &LateContext<'_>, item: &ImplItem<'_>) -> bool {
    match item.kind {
        ImplItemKind::Fn(_, eid) => is_relevant_expr(cx, cx.tcx.typeck_body(eid), &cx.tcx.hir().body(eid).value),
        _ => false,
    }
}

fn is_relevant_trait(cx: &LateContext<'_>, item: &TraitItem<'_>) -> bool {
    match item.kind {
        TraitItemKind::Fn(_, TraitFn::Required(_)) => true,
        TraitItemKind::Fn(_, TraitFn::Provided(eid)) => {
            is_relevant_expr(cx, cx.tcx.typeck_body(eid), &cx.tcx.hir().body(eid).value)
        },
        _ => false,
    }
}

fn is_relevant_block(cx: &LateContext<'_>, typeck_results: &ty::TypeckResults<'_>, block: &Block<'_>) -> bool {
    block.stmts.first().map_or(
        block
            .expr
            .as_ref()
            .map_or(false, |e| is_relevant_expr(cx, typeck_results, e)),
        |stmt| match &stmt.kind {
            StmtKind::Local(_) => true,
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => is_relevant_expr(cx, typeck_results, expr),
            _ => false,
        },
    )
}

fn is_relevant_expr(cx: &LateContext<'_>, typeck_results: &ty::TypeckResults<'_>, expr: &Expr<'_>) -> bool {
    match &expr.kind {
        ExprKind::Block(block, _) => is_relevant_block(cx, typeck_results, block),
        ExprKind::Ret(Some(e)) => is_relevant_expr(cx, typeck_results, e),
        ExprKind::Ret(None) | ExprKind::Break(_, None) => false,
        ExprKind::Call(path_expr, _) => {
            if let ExprKind::Path(qpath) = &path_expr.kind {
                typeck_results
                    .qpath_res(qpath, path_expr.hir_id)
                    .opt_def_id()
                    .map_or(true, |fun_id| !match_def_path(cx, fun_id, &paths::BEGIN_PANIC))
            } else {
                true
            }
        },
        _ => true,
    }
}

fn check_attrs(cx: &LateContext<'_>, span: Span, name: Symbol, attrs: &[Attribute]) {
    if span.from_expansion() {
        return;
    }

    for attr in attrs {
        if let Some(values) = attr.meta_item_list() {
            if values.len() != 1 || !attr.has_name(sym!(inline)) {
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

fn check_semver(cx: &LateContext<'_>, span: Span, lit: &Lit) {
    if let LitKind::Str(is, _) = lit.kind {
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
        mi.is_word() && mi.has_name(expected)
    } else {
        false
    }
}

declare_lint_pass!(EarlyAttributes => [
    DEPRECATED_CFG_ATTR,
    MISMATCHED_TARGET_OS,
    EMPTY_LINE_AFTER_OUTER_ATTR,
]);

impl EarlyLintPass for EarlyAttributes {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &rustc_ast::Item) {
        check_empty_line_after_outer_attr(cx, item);
    }

    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &Attribute) {
        check_deprecated_cfg_attr(cx, attr);
        check_mismatched_target_os(cx, attr);
    }
}

fn check_empty_line_after_outer_attr(cx: &EarlyContext<'_>, item: &rustc_ast::Item) {
    for attr in &item.attrs {
        let attr_item = if let AttrKind::Normal(ref attr) = attr.kind {
            attr
        } else {
            return;
        };

        if attr.style == AttrStyle::Outer {
            if attr_item.args.inner_tokens().is_empty() || !is_present_in_source(cx, attr.span) {
                return;
            }

            let begin_of_attr_to_item = Span::new(attr.span.lo(), item.span.lo(), item.span.ctxt());
            let end_of_attr_to_item = Span::new(attr.span.hi(), item.span.lo(), item.span.ctxt());

            if let Some(snippet) = snippet_opt(cx, end_of_attr_to_item) {
                let lines = snippet.split('\n').collect::<Vec<_>>();
                let lines = without_block_comments(lines);

                if lines.iter().filter(|l| l.trim().is_empty()).count() > 2 {
                    span_lint(
                        cx,
                        EMPTY_LINE_AFTER_OUTER_ATTR,
                        begin_of_attr_to_item,
                        "found an empty line after an outer attribute. \
                        Perhaps you forgot to add a `!` to make it an inner attribute?",
                    );
                }
            }
        }
    }
}

fn check_deprecated_cfg_attr(cx: &EarlyContext<'_>, attr: &Attribute) {
    if_chain! {
        // check cfg_attr
        if attr.has_name(sym!(cfg_attr));
        if let Some(items) = attr.meta_item_list();
        if items.len() == 2;
        // check for `rustfmt`
        if let Some(feature_item) = items[0].meta_item();
        if feature_item.has_name(sym!(rustfmt));
        // check for `rustfmt_skip` and `rustfmt::skip`
        if let Some(skip_item) = &items[1].meta_item();
        if skip_item.has_name(sym!(rustfmt_skip)) ||
            skip_item.path.segments.last().expect("empty path in attribute").ident.name == sym!(skip);
        // Only lint outer attributes, because custom inner attributes are unstable
        // Tracking issue: https://github.com/rust-lang/rust/issues/54726
        if let AttrStyle::Outer = attr.style;
        then {
            span_lint_and_sugg(
                cx,
                DEPRECATED_CFG_ATTR,
                attr.span,
                "`cfg_attr` is deprecated for rustfmt and got replaced by tool attributes",
                "use",
                "#[rustfmt::skip]".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn check_mismatched_target_os(cx: &EarlyContext<'_>, attr: &Attribute) {
    fn find_os(name: &str) -> Option<&'static str> {
        UNIX_SYSTEMS
            .iter()
            .chain(NON_UNIX_SYSTEMS.iter())
            .find(|&&os| os == name)
            .copied()
    }

    fn is_unix(name: &str) -> bool {
        UNIX_SYSTEMS.iter().any(|&os| os == name)
    }

    fn find_mismatched_target_os(items: &[NestedMetaItem]) -> Vec<(&str, Span)> {
        let mut mismatched = Vec::new();

        for item in items {
            if let NestedMetaItem::MetaItem(meta) = item {
                match &meta.kind {
                    MetaItemKind::List(list) => {
                        mismatched.extend(find_mismatched_target_os(&list));
                    },
                    MetaItemKind::Word => {
                        if_chain! {
                            if let Some(ident) = meta.ident();
                            if let Some(os) = find_os(&*ident.name.as_str());
                            then {
                                mismatched.push((os, ident.span));
                            }
                        }
                    },
                    _ => {},
                }
            }
        }

        mismatched
    }

    if_chain! {
        if attr.has_name(sym!(cfg));
        if let Some(list) = attr.meta_item_list();
        let mismatched = find_mismatched_target_os(&list);
        if !mismatched.is_empty();
        then {
            let mess = "operating system used in target family position";

            span_lint_and_then(cx, MISMATCHED_TARGET_OS, attr.span, &mess, |diag| {
                // Avoid showing the unix suggestion multiple times in case
                // we have more than one mismatch for unix-like systems
                let mut unix_suggested = false;

                for (os, span) in mismatched {
                    let sugg = format!("target_os = \"{}\"", os);
                    diag.span_suggestion(span, "try", sugg, Applicability::MaybeIncorrect);

                    if !unix_suggested && is_unix(os) {
                        diag.help("Did you mean `unix`?");
                        unix_suggested = true;
                    }
                }
            });
        }
    }
}
