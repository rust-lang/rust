mod allow_attributes;
mod allow_attributes_without_reason;
mod blanket_clippy_restriction_lints;
mod deprecated_cfg_attr;
mod deprecated_semver;
mod duplicated_attributes;
mod inline_always;
mod mixed_attributes_style;
mod non_minimal_cfg;
mod repr_attributes;
mod should_panic_without_expect;
mod unnecessary_clippy_cfg;
mod useless_attribute;
mod utils;

use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::msrvs::{self, Msrv, MsrvStack};
use rustc_ast::{self as ast, AttrArgs, AttrKind, Attribute, MetaItemInner, MetaItemKind};
use rustc_hir::{ImplItem, Item, ItemKind, TraitItem};
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
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
    /// * ambiguous_glob_reexports
    /// * dead_code
    /// * deprecated
    /// * hidden_glob_reexports
    /// * unreachable_pub
    /// * unused
    /// * unused_braces
    /// * unused_import_braces
    /// * clippy::disallowed_types
    /// * clippy::enum_glob_use
    /// * clippy::macro_use_imports
    /// * clippy::module_name_repetitions
    /// * clippy::redundant_pub_crate
    /// * clippy::single_component_path_imports
    /// * clippy::unsafe_removed_from_name
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
    /// Checks for attributes that allow lints without a reason.
    ///
    /// ### Why restrict this?
    /// Justifying each `allow` helps readers understand the reasoning,
    /// and may allow removing `allow` attributes if their purpose is obsolete.
    ///
    /// ### Example
    /// ```no_run
    /// #![allow(clippy::some_lint)]
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #![allow(clippy::some_lint, reason = "False positive rust-lang/rust-clippy#1002020")]
    /// ```
    #[clippy::version = "1.61.0"]
    pub ALLOW_ATTRIBUTES_WITHOUT_REASON,
    restriction,
    "ensures that all `allow` and `expect` attributes have a reason"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of the `#[allow]` attribute and suggests replacing it with
    /// the `#[expect]` (See [RFC 2383](https://rust-lang.github.io/rfcs/2383-lint-reasons.html))
    ///
    /// This lint only warns outer attributes (`#[allow]`), as inner attributes
    /// (`#![allow]`) are usually used to enable or disable lints on a global scale.
    ///
    /// ### Why is this bad?
    /// `#[expect]` attributes suppress the lint emission, but emit a warning, if
    /// the expectation is unfulfilled. This can be useful to be notified when the
    /// lint is no longer triggered.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[allow(unused_mut)]
    /// fn foo() -> usize {
    ///     let mut a = Vec::new();
    ///     a.len()
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// #[expect(unused_mut)]
    /// fn foo() -> usize {
    ///     let mut a = Vec::new();
    ///     a.len()
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub ALLOW_ATTRIBUTES,
    restriction,
    "`#[allow]` will not trigger if a warning isn't found. `#[expect]` triggers if there are no warnings."
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
    /// Checks for items with `#[repr(packed)]`-attribute without ABI qualification
    ///
    /// ### Why is this bad?
    /// Without qualification, `repr(packed)` implies `repr(Rust)`. The Rust-ABI is inherently unstable.
    /// While this is fine as long as the type is accessed correctly within Rust-code, most uses
    /// of `#[repr(packed)]` involve FFI and/or data structures specified by network-protocols or
    /// other external specifications. In such situations, the unstable Rust-ABI implied in
    /// `#[repr(packed)]` may lead to future bugs should the Rust-ABI change.
    ///
    /// In case you are relying on a well defined and stable memory layout, qualify the type's
    /// representation using the `C`-ABI. Otherwise, if the type in question is only ever
    /// accessed from Rust-code according to Rust's rules, use the `Rust`-ABI explicitly.
    ///
    /// ### Example
    /// ```no_run
    /// #[repr(packed)]
    /// struct NetworkPacketHeader {
    ///     header_length: u8,
    ///     header_version: u16
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[repr(C, packed)]
    /// struct NetworkPacketHeader {
    ///     header_length: u8,
    ///     header_version: u16
    /// }
    /// ```
    #[clippy::version = "1.85.0"]
    pub REPR_PACKED_WITHOUT_ABI,
    suspicious,
    "ensures that `repr(packed)` always comes with a qualified ABI"
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
    /// Checks for items that have the same kind of attributes with mixed styles (inner/outer).
    ///
    /// ### Why is this bad?
    /// Having both style of said attributes makes it more complicated to read code.
    ///
    /// ### Known problems
    /// This lint currently has false-negatives when mixing same attributes
    /// but they have different path symbols, for example:
    /// ```ignore
    /// #[custom_attribute]
    /// pub fn foo() {
    ///     #![my_crate::custom_attribute]
    /// }
    /// ```
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
    style,
    "item has both inner and outer attributes"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for attributes that appear two or more times.
    ///
    /// ### Why is this bad?
    /// Repeating an attribute on the same item (or globally on the same crate)
    /// is unnecessary and doesn't have an effect.
    ///
    /// ### Example
    /// ```no_run
    /// #[allow(dead_code)]
    /// #[allow(dead_code)]
    /// fn foo() {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[allow(dead_code)]
    /// fn foo() {}
    /// ```
    #[clippy::version = "1.79.0"]
    pub DUPLICATED_ATTRIBUTES,
    suspicious,
    "duplicated attribute"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for ignored tests without messages.
    ///
    /// ### Why is this bad?
    /// The reason for ignoring the test may not be obvious.
    ///
    /// ### Example
    /// ```no_run
    /// #[test]
    /// #[ignore]
    /// fn test() {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[test]
    /// #[ignore = "Some good reason"]
    /// fn test() {}
    /// ```
    #[clippy::version = "1.85.0"]
    pub IGNORE_WITHOUT_REASON,
    pedantic,
    "ignored tests without messages"
}

pub struct Attributes {
    msrv: Msrv,
}

impl_lint_pass!(Attributes => [
    INLINE_ALWAYS,
    REPR_PACKED_WITHOUT_ABI,
]);

impl Attributes {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for Attributes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let attrs = cx.tcx.hir_attrs(item.hir_id());
        if let ItemKind::Fn { ident, .. } = item.kind
            && is_relevant_item(cx, item)
        {
            inline_always::check(cx, item.span, ident.name, attrs);
        }
        repr_attributes::check(cx, item.span, attrs, self.msrv);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if is_relevant_impl(cx, item) {
            inline_always::check(cx, item.span, item.ident.name, cx.tcx.hir_attrs(item.hir_id()));
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if is_relevant_trait(cx, item) {
            inline_always::check(cx, item.span, item.ident.name, cx.tcx.hir_attrs(item.hir_id()));
        }
    }
}

pub struct EarlyAttributes {
    msrv: MsrvStack,
}

impl EarlyAttributes {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: MsrvStack::new(conf.msrv),
        }
    }
}

impl_lint_pass!(EarlyAttributes => [
    DEPRECATED_CFG_ATTR,
    NON_MINIMAL_CFG,
    DEPRECATED_CLIPPY_CFG_ATTR,
    UNNECESSARY_CLIPPY_CFG,
]);

impl EarlyLintPass for EarlyAttributes {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &Attribute) {
        deprecated_cfg_attr::check(cx, attr, &self.msrv);
        deprecated_cfg_attr::check_clippy(cx, attr);
        non_minimal_cfg::check(cx, attr);
    }

    extract_msrv_attr!();
}

pub struct PostExpansionEarlyAttributes {
    msrv: MsrvStack,
}

impl PostExpansionEarlyAttributes {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: MsrvStack::new(conf.msrv),
        }
    }
}

impl_lint_pass!(PostExpansionEarlyAttributes => [
    ALLOW_ATTRIBUTES,
    ALLOW_ATTRIBUTES_WITHOUT_REASON,
    DEPRECATED_SEMVER,
    IGNORE_WITHOUT_REASON,
    USELESS_ATTRIBUTE,
    BLANKET_CLIPPY_RESTRICTION_LINTS,
    SHOULD_PANIC_WITHOUT_EXPECT,
    MIXED_ATTRIBUTES_STYLE,
    DUPLICATED_ATTRIBUTES,
]);

impl EarlyLintPass for PostExpansionEarlyAttributes {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &ast::Crate) {
        blanket_clippy_restriction_lints::check_command_line(cx);
        duplicated_attributes::check(cx, &krate.attrs);
    }

    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &Attribute) {
        if let Some(items) = &attr.meta_item_list()
            && let Some(ident) = attr.ident()
        {
            if matches!(ident.name, sym::allow) && self.msrv.meets(msrvs::LINT_REASONS_STABILIZATION) {
                allow_attributes::check(cx, attr);
            }
            if matches!(ident.name, sym::allow | sym::expect) && self.msrv.meets(msrvs::LINT_REASONS_STABILIZATION) {
                allow_attributes_without_reason::check(cx, ident.name, items, attr);
            }
            if is_lint_level(ident.name, attr.id) {
                blanket_clippy_restriction_lints::check(cx, ident.name, items);
            }
            if items.is_empty() || !attr.has_name(sym::deprecated) {
                return;
            }
            for item in items {
                if let MetaItemInner::MetaItem(mi) = &item
                    && let MetaItemKind::NameValue(lit) = &mi.kind
                    && mi.has_name(sym::since)
                {
                    deprecated_semver::check(cx, item.span(), lit);
                }
            }
        }

        if attr.has_name(sym::should_panic) {
            should_panic_without_expect::check(cx, attr);
        }

        if attr.has_name(sym::ignore)
            && match &attr.kind {
                AttrKind::Normal(normal_attr) => !matches!(normal_attr.item.args, AttrArgs::Eq { .. }),
                AttrKind::DocComment(..) => true,
            }
        {
            span_lint_and_help(
                cx,
                IGNORE_WITHOUT_REASON,
                attr.span,
                "`#[ignore]` without reason",
                None,
                "add a reason with `= \"..\"`",
            );
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &'_ ast::Item) {
        match item.kind {
            ast::ItemKind::ExternCrate(..) | ast::ItemKind::Use(..) => useless_attribute::check(cx, item, &item.attrs),
            _ => {},
        }

        mixed_attributes_style::check(cx, item.span, &item.attrs);
        duplicated_attributes::check(cx, &item.attrs);
    }

    extract_msrv_attr!();
}
