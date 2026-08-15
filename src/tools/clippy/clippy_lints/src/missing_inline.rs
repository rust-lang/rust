use clippy_utils::diagnostics::span_lint;
use rustc_hir::{ImplItem, ImplItemKind, Item, ItemKind, OwnerId, TraitFn, TraitItem, TraitItemKind, find_attr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::config::CrateType;
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// It lints if an exported function, method, trait method with default impl,
    /// or trait method impl is not `#[inline]`.
    ///
    /// ### Why restrict this?
    /// When a function is not marked `#[inline]`, it is not
    /// [a “small” candidate for automatic inlining][small], and LTO is not in use, then it is not
    /// possible for the function to be inlined into the code of any crate other than the one in
    /// which it is defined.  Depending on the role of the function and the relationship of the crates,
    /// this could significantly reduce performance.
    ///
    /// Certain types of crates might intend for most of the methods in their public API to be able
    /// to be inlined across crates even when LTO is disabled.
    /// This lint allows those crates to require all exported methods to be `#[inline]` by default, and
    /// then opt out for specific methods where this might not make sense.
    ///
    /// ### Example
    /// ```no_run
    /// pub fn foo() {} // missing #[inline]
    /// fn ok() {} // ok
    /// #[inline] pub fn bar() {} // ok
    /// #[inline(always)] pub fn baz() {} // ok
    ///
    /// pub trait Bar {
    ///   fn bar(); // ok
    ///   fn def_bar() {} // missing #[inline]
    /// }
    ///
    /// struct Baz;
    /// impl Baz {
    ///     fn private() {} // ok
    /// }
    ///
    /// impl Bar for Baz {
    ///   fn bar() {} // ok - Baz is not exported
    /// }
    ///
    /// pub struct PubBaz;
    /// impl PubBaz {
    ///     fn private() {} // ok
    ///     pub fn not_private() {} // missing #[inline]
    /// }
    ///
    /// impl Bar for PubBaz {
    ///     fn bar() {} // missing #[inline]
    ///     fn def_bar() {} // missing #[inline]
    /// }
    /// ```
    ///
    /// [small]: https://github.com/rust-lang/rust/pull/116505
    #[clippy::version = "pre 1.29.0"]
    pub MISSING_INLINE_IN_PUBLIC_ITEMS,
    restriction,
    "detects missing `#[inline]` attribute for public callables (functions, trait methods, methods...)"
}

declare_lint_pass!(MissingInline => [MISSING_INLINE_IN_PUBLIC_ITEMS]);

fn check(cx: &LateContext<'_>, item: OwnerId, sp: Span) {
    if cx.effective_visibilities.is_exported(item.def_id)
        && !find_attr!(cx.tcx.hir_attrs(item.into()), Inline(..))
        // Rust `inline` doesn't mean anything with external linkage.
        && !cx.tcx.codegen_fn_attrs(item.def_id).contains_extern_indicator()
        && !cx.tcx.crate_types().iter().any(|&t| matches!(t, CrateType::ProcMacro))
        && !sp.in_external_macro(cx.tcx.sess.source_map())
    {
        span_lint(
            cx,
            MISSING_INLINE_IN_PUBLIC_ITEMS,
            sp,
            "missing `#[inline]` on a publicly callable function",
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for MissingInline {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx Item<'_>) {
        if let ItemKind::Fn { .. } = it.kind {
            check(cx, it.owner_id, it.span);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(_, f) = item.kind
            && let TraitFn::Provided(_) = f
        {
            check(cx, item.owner_id, item.span);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Fn(..) = item.kind {
            check(cx, item.owner_id, item.span);
        }
    }
}
