//! checks for attributes

mod allow_attributes_without_reason;
mod blanket_clippy_restriction_lints;
mod deprecated_cfg_attr;
mod deprecated_semver;
mod empty_line_after;
mod inline_always;
mod maybe_misused_cfg;
mod mismatched_target_os;
mod mixed_attributes_style;
mod non_minimal_cfg;
mod should_panic_without_expect;
mod unnecessary_clippy_cfg;
mod useless_attribute;
mod utils;

use clippy_config::msrvs::Msrv;
use rustc_ast::{Attribute, MetaItemKind, NestedMetaItem};
use rustc_hir::{ImplItem, Item, ItemKind, TraitItem};
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, impl_lint_pass};
use rustc_span::sym;
use utils::{is_lint_level, is_relevant_impl, is_relevant_item, is_relevant_trait};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for items annotated with `#[inline(always)]`,
    /// unless the annotated function is empty or simply panics.
    ///
    /// ### Why is this bad?
    /// While there are valid uses of this annotation (and once
    /// you know when to use it, by all means `allow` this lint), it's a common
    /// newbie-mistake to pepper one's code with it.
    ///
    /// As a rule of thumb, before slapping `#[inline(always)]` on a function,
    /// measure if that additional function call really affects your runtime profile
    /// sufficiently to make up for the increase in compile time.
    ///
    /// ### Known problems
    /// False positives, big time. This lint is meant to be
    /// deactivated by everyone doing serious performance work. This means having
    /// done the measurement.
    ///
    /// ### Example
    /// ```ignore
    /// #[inline(always)]
    /// fn not_quite_hot_code(..) { ... }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INLINE_ALWAYS,
    pedantic,
    "use of `#[inline(always)]`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `extern crate` and `use` items annotated with
    /// lint attributes.
    ///
    /// This lint permits lint attributes for lints emitted on the items themself.
    /// For `use` items these lints are:
    /// * deprecated
    /// * unreachable_pub
    /// * unused_imports
    /// * clippy::enum_glob_use
    /// * clippy::macro_use_imports
    /// * clippy::wildcard_imports
    ///
    /// For `extern crate` items these lints are:
    /// * `unused_imports` on items with `#[macro_use]`
    ///
    /// ### Why is this bad?
    /// Lint attributes have no effect on crate imports. Most
    /// likely a `!` was forgotten.
    ///
    /// ### Example
    /// ```ignore
    /// #[deny(dead_code)]
    /// extern crate foo;
    /// #[forbid(dead_code)]
    /// use foo::bar;
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// #[allow(unused_imports)]
    /// use foo::baz;
    /// #[allow(unused_imports)]
    /// #[macro_use]
    /// extern crate baz;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_ATTRIBUTE,
    correctness,
    "use of lint attributes on `extern crate` items"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[deprecated]` annotations with a `since`
    /// field that is not a valid semantic version. Also allows "TBD" to signal
    /// future deprecation.
    ///
    /// ### Why is this bad?
    /// For checking the version of the deprecation, it must be
    /// a valid semver. Failing that, the contained information is useless.
    ///
    /// ### Example
    /// ```no_run
    /// #[deprecated(since = "forever")]
    /// fn something_else() { /* ... */ }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DEPRECATED_SEMVER,
    correctness,
    "use of `#[deprecated(since = \"x\")]` where x is not semver"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty lines after outer attributes
    ///
    /// ### Why is this bad?
    /// Most likely the attribute was meant to be an inner attribute using a '!'.
    /// If it was meant to be an outer attribute, then the following item
    /// should not be separated by empty lines.
    ///
    /// ### Known problems
    /// Can cause false positives.
    ///
    /// From the clippy side it's difficult to detect empty lines between an attributes and the
    /// following item because empty lines and comments are not part of the AST. The parsing
    /// currently works for basic cases but is not perfect.
    ///
    /// ### Example
    /// ```no_run
    /// #[allow(dead_code)]
    ///
    /// fn not_quite_good_code() { }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// // Good (as inner attribute)
    /// #![allow(dead_code)]
    ///
    /// fn this_is_fine() { }
    ///
    /// // or
    ///
    /// // Good (as outer attribute)
    /// #[allow(dead_code)]
    /// fn this_is_fine_too() { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EMPTY_LINE_AFTER_OUTER_ATTR,
    nursery,
    "empty line after outer attribute"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty lines after documentation comments.
    ///
    /// ### Why is this bad?
    /// The documentation comment was most likely meant to be an inner attribute or regular comment.
    /// If it was intended to be a documentation comment, then the empty line should be removed to
    /// be more idiomatic.
    ///
    /// ### Known problems
    /// Only detects empty lines immediately following the documentation. If the doc comment is followed
    /// by an attribute and then an empty line, this lint will not trigger. Use `empty_line_after_outer_attr`
    /// in combination with this lint to detect both cases.
    ///
    /// Does not detect empty lines after doc attributes (e.g. `#[doc = ""]`).
    ///
    /// ### Example
    /// ```no_run
    /// /// Some doc comment with a blank line after it.
    ///
    /// fn not_quite_good_code() { }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// /// Good (no blank line)
    /// fn this_is_fine() { }
    /// ```
    ///
    /// ```no_run
    /// // Good (convert to a regular comment)
    ///
    /// fn this_is_fine_too() { }
    /// ```
    ///
    /// ```no_run
    /// //! Good (convert to a comment on an inner attribute)
    ///
    /// fn this_is_fine_as_well() { }
    /// ```
    #[clippy::version = "1.70.0"]
    pub EMPTY_LINE_AFTER_DOC_COMMENTS,
    nursery,
    "empty line after documentation comments"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `warn`/`deny`/`forbid` attributes targeting the whole clippy::restriction category.
    ///
    /// ### Why is this bad?
    /// Restriction lints sometimes are in contrast with other lints or even go against idiomatic rust.
    /// These lints should only be enabled on a lint-by-lint basis and with careful consideration.
    ///
    /// ### Example
    /// ```no_run
    /// #![deny(clippy::restriction)]
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #![deny(clippy::as_conversions)]
    /// ```
    #[clippy::version = "1.47.0"]
    pub BLANKET_CLIPPY_RESTRICTION_LINTS,
    suspicious,
    "enabling the complete restriction group"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[cfg_attr(rustfmt, rustfmt_skip)]` and suggests to replace it
    /// with `#[rustfmt::skip]`.
    ///
    /// ### Why is this bad?
    /// Since tool_attributes ([rust-lang/rust#44690](https://github.com/rust-lang/rust/issues/44690))
    /// are stable now, they should be used instead of the old `cfg_attr(rustfmt)` attributes.
    ///
    /// ### Known problems
    /// This lint doesn't detect crate level inner attributes, because they get
    /// processed before the PreExpansionPass lints get executed. See
    /// [#3123](https://github.com/rust-lang/rust-clippy/pull/3123#issuecomment-422321765)
    ///
    /// ### Example
    /// ```no_run
    /// #[cfg_attr(rustfmt, rustfmt_skip)]
    /// fn main() { }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[rustfmt::skip]
    /// fn main() { }
    /// ```
    #[clippy::version = "1.32.0"]
    pub DEPRECATED_CFG_ATTR,
    complexity,
    "usage of `cfg_attr(rustfmt)` instead of tool attributes"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for cfg attributes having operating systems used in target family position.
    ///
    /// ### Why is this bad?
    /// The configuration option will not be recognised and the related item will not be included
    /// by the conditional compilation engine.
    ///
    /// ### Example
    /// ```no_run
    /// #[cfg(linux)]
    /// fn conditional() { }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # mod hidden {
    /// #[cfg(target_os = "linux")]
    /// fn conditional() { }
    /// # }
    ///
    /// // or
    ///
    /// #[cfg(unix)]
    /// fn conditional() { }
    /// ```
    /// Check the [Rust Reference](https://doc.rust-lang.org/reference/conditional-compilation.html#target_os) for more details.
    #[clippy::version = "1.45.0"]
    pub MISMATCHED_TARGET_OS,
    correctness,
    "usage of `cfg(operating_system)` instead of `cfg(target_os = \"operating_system\")`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for attributes that allow lints without a reason.
    ///
    /// (This requires the `lint_reasons` feature)
    ///
    /// ### Why is this bad?
    /// Allowing a lint should always have a reason. This reason should be documented to
    /// ensure that others understand the reasoning
    ///
    /// ### Example
    /// ```no_run
    /// #![feature(lint_reasons)]
    ///
    /// #![allow(clippy::some_lint)]
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #![feature(lint_reasons)]
    ///
    /// #![allow(clippy::some_lint, reason = "False positive rust-lang/rust-clippy#1002020")]
    /// ```
    #[clippy::version = "1.61.0"]
    pub ALLOW_ATTRIBUTES_WITHOUT_REASON,
    restriction,
    "ensures that all `allow` and `expect` attributes have a reason"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[should_panic]` attributes without specifying the expected panic message.
    ///
    /// ### Why is this bad?
    /// The expected panic message should be specified to ensure that the test is actually
    /// panicking with the expected message, and not another unrelated panic.
    ///
    /// ### Example
    /// ```no_run
    /// fn random() -> i32 { 0 }
    ///
    /// #[should_panic]
    /// #[test]
    /// fn my_test() {
    ///     let _ = 1 / random();
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn random() -> i32 { 0 }
    ///
    /// #[should_panic = "attempt to divide by zero"]
    /// #[test]
    /// fn my_test() {
    ///     let _ = 1 / random();
    /// }
    /// ```
    #[clippy::version = "1.74.0"]
    pub SHOULD_PANIC_WITHOUT_EXPECT,
    pedantic,
    "ensures that all `should_panic` attributes specify its expected panic message"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `any` and `all` combinators in `cfg` with only one condition.
    ///
    /// ### Why is this bad?
    /// If there is only one condition, no need to wrap it into `any` or `all` combinators.
    ///
    /// ### Example
    /// ```no_run
    /// #[cfg(any(unix))]
    /// pub struct Bar;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[cfg(unix)]
    /// pub struct Bar;
    /// ```
    #[clippy::version = "1.71.0"]
    pub NON_MINIMAL_CFG,
    style,
    "ensure that all `cfg(any())` and `cfg(all())` have more than one condition"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[cfg(features = "...")]` and suggests to replace it with
    /// `#[cfg(feature = "...")]`.
    ///
    /// It also checks if `cfg(test)` was misspelled.
    ///
    /// ### Why is this bad?
    /// Misspelling `feature` as `features` or `test` as `tests` can be sometimes hard to spot. It
    /// may cause conditional compilation not work quietly.
    ///
    /// ### Example
    /// ```no_run
    /// #[cfg(features = "some-feature")]
    /// fn conditional() { }
    /// #[cfg(tests)]
    /// mod tests { }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[cfg(feature = "some-feature")]
    /// fn conditional() { }
    /// #[cfg(test)]
    /// mod tests { }
    /// ```
    #[clippy::version = "1.69.0"]
    pub MAYBE_MISUSED_CFG,
    suspicious,
    "prevent from misusing the wrong attr name"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[cfg_attr(feature = "cargo-clippy", ...)]` and for
    /// `#[cfg(feature = "cargo-clippy")]` and suggests to replace it with
    /// `#[cfg_attr(clippy, ...)]` or `#[cfg(clippy)]`.
    ///
    /// ### Why is this bad?
    /// This feature has been deprecated for years and shouldn't be used anymore.
    ///
    /// ### Example
    /// ```no_run
    /// #[cfg(feature = "cargo-clippy")]
    /// struct Bar;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[cfg(clippy)]
    /// struct Bar;
    /// ```
    #[clippy::version = "1.78.0"]
    pub DEPRECATED_CLIPPY_CFG_ATTR,
    suspicious,
    "usage of `cfg(feature = \"cargo-clippy\")` instead of `cfg(clippy)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[cfg_attr(clippy, allow(clippy::lint))]`
    /// and suggests to replace it with `#[allow(clippy::lint)]`.
    ///
    /// ### Why is this bad?
    /// There is no reason to put clippy attributes behind a clippy `cfg` as they are not
    /// run by anything else than clippy.
    ///
    /// ### Example
    /// ```no_run
    /// #![cfg_attr(clippy, allow(clippy::deprecated_cfg_attr))]
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #![allow(clippy::deprecated_cfg_attr)]
    /// ```
    #[clippy::version = "1.78.0"]
    pub UNNECESSARY_CLIPPY_CFG,
    suspicious,
    "usage of `cfg_attr(clippy, allow(clippy::lint))` instead of `allow(clippy::lint)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks that an item has only one kind of attributes.
    ///
    /// ### Why is this bad?
    /// Having both kinds of attributes makes it more complicated to read code.
    ///
    /// ### Example
    /// ```no_run
    /// #[cfg(linux)]
    /// pub fn foo() {
    ///     #![cfg(windows)]
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[cfg(linux)]
    /// #[cfg(windows)]
    /// pub fn foo() {
    /// }
    /// ```
    #[clippy::version = "1.78.0"]
    pub MIXED_ATTRIBUTES_STYLE,
    suspicious,
    "item has both inner and outer attributes"
}

declare_lint_pass!(Attributes => [
    ALLOW_ATTRIBUTES_WITHOUT_REASON,
    INLINE_ALWAYS,
    DEPRECATED_SEMVER,
    USELESS_ATTRIBUTE,
    BLANKET_CLIPPY_RESTRICTION_LINTS,
    SHOULD_PANIC_WITHOUT_EXPECT,
]);

impl<'tcx> LateLintPass<'tcx> for Attributes {
    fn check_crate(&mut self, cx: &LateContext<'tcx>) {
        blanket_clippy_restriction_lints::check_command_line(cx);
    }

    fn check_attribute(&mut self, cx: &LateContext<'tcx>, attr: &'tcx Attribute) {
        if let Some(items) = &attr.meta_item_list() {
            if let Some(ident) = attr.ident() {
                if is_lint_level(ident.name, attr.id) {
                    blanket_clippy_restriction_lints::check(cx, ident.name, items);
                }
                if matches!(ident.name, sym::allow | sym::expect) {
                    allow_attributes_without_reason::check(cx, ident.name, items, attr);
                }
                if items.is_empty() || !attr.has_name(sym::deprecated) {
                    return;
                }
                for item in items {
                    if let NestedMetaItem::MetaItem(mi) = &item
                        && let MetaItemKind::NameValue(lit) = &mi.kind
                        && mi.has_name(sym::since)
                    {
                        deprecated_semver::check(cx, item.span(), lit);
                    }
                }
            }
        }
        if attr.has_name(sym::should_panic) {
            should_panic_without_expect::check(cx, attr);
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let attrs = cx.tcx.hir().attrs(item.hir_id());
        if is_relevant_item(cx, item) {
            inline_always::check(cx, item.span, item.ident.name, attrs);
        }
        match item.kind {
            ItemKind::ExternCrate(..) | ItemKind::Use(..) => useless_attribute::check(cx, item, attrs),
            _ => {},
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if is_relevant_impl(cx, item) {
            inline_always::check(cx, item.span, item.ident.name, cx.tcx.hir().attrs(item.hir_id()));
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if is_relevant_trait(cx, item) {
            inline_always::check(cx, item.span, item.ident.name, cx.tcx.hir().attrs(item.hir_id()));
        }
    }
}

pub struct EarlyAttributes {
    pub msrv: Msrv,
}

impl_lint_pass!(EarlyAttributes => [
    DEPRECATED_CFG_ATTR,
    MISMATCHED_TARGET_OS,
    EMPTY_LINE_AFTER_OUTER_ATTR,
    EMPTY_LINE_AFTER_DOC_COMMENTS,
    NON_MINIMAL_CFG,
    MAYBE_MISUSED_CFG,
    DEPRECATED_CLIPPY_CFG_ATTR,
    UNNECESSARY_CLIPPY_CFG,
    MIXED_ATTRIBUTES_STYLE,
]);

impl EarlyLintPass for EarlyAttributes {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &rustc_ast::Item) {
        empty_line_after::check(cx, item);
        mixed_attributes_style::check(cx, item);
    }

    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &Attribute) {
        deprecated_cfg_attr::check(cx, attr, &self.msrv);
        deprecated_cfg_attr::check_clippy(cx, attr);
        mismatched_target_os::check(cx, attr);
        non_minimal_cfg::check(cx, attr);
        maybe_misused_cfg::check(cx, attr);
    }

    extract_msrv_attr!(EarlyContext);
}
