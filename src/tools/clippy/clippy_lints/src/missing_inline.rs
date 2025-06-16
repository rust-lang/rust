use clippy_utils::diagnostics::span_lint;
use rustc_attr_data_structures::{find_attr, AttributeKind};
use rustc_hir as hir;
use rustc_hir::Attribute;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::AssocItemContainer;
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

fn check_missing_inline_attrs(cx: &LateContext<'_>, attrs: &[Attribute], sp: Span, desc: &'static str) {
    if !find_attr!(attrs, AttributeKind::Inline(..)) {
        span_lint(
            cx,
            MISSING_INLINE_IN_PUBLIC_ITEMS,
            sp,
            format!("missing `#[inline]` for {desc}"),
        );
    }
}

fn is_executable_or_proc_macro(cx: &LateContext<'_>) -> bool {
    use rustc_session::config::CrateType;

    cx.tcx
        .crate_types()
        .iter()
        .any(|t: &CrateType| matches!(t, CrateType::Executable | CrateType::ProcMacro))
}

declare_lint_pass!(MissingInline => [MISSING_INLINE_IN_PUBLIC_ITEMS]);

impl<'tcx> LateLintPass<'tcx> for MissingInline {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::Item<'_>) {
        if it.span.in_external_macro(cx.sess().source_map()) || is_executable_or_proc_macro(cx) {
            return;
        }

        if !cx.effective_visibilities.is_exported(it.owner_id.def_id) {
            return;
        }
        match it.kind {
            hir::ItemKind::Fn { .. } => {
                let desc = "a function";
                let attrs = cx.tcx.hir_attrs(it.hir_id());
                check_missing_inline_attrs(cx, attrs, it.span, desc);
            },
            hir::ItemKind::Trait(ref _is_auto, ref _unsafe, _ident, _generics, _bounds, trait_items) => {
                // note: we need to check if the trait is exported so we can't use
                // `LateLintPass::check_trait_item` here.
                for tit in trait_items {
                    let tit_ = cx.tcx.hir_trait_item(tit.id);
                    match tit_.kind {
                        hir::TraitItemKind::Const(..) | hir::TraitItemKind::Type(..) => {},
                        hir::TraitItemKind::Fn(..) => {
                            if cx.tcx.defaultness(tit.id.owner_id).has_value() {
                                // trait method with default body needs inline in case
                                // an impl is not provided
                                let desc = "a default trait method";
                                let item = cx.tcx.hir_trait_item(tit.id);
                                let attrs = cx.tcx.hir_attrs(item.hir_id());
                                check_missing_inline_attrs(cx, attrs, item.span, desc);
                            }
                        },
                    }
                }
            },
            hir::ItemKind::Const(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::Static(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::GlobalAsm { .. }
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::Impl { .. }
            | hir::ItemKind::Use(..) => {},
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx hir::ImplItem<'_>) {
        if impl_item.span.in_external_macro(cx.sess().source_map()) || is_executable_or_proc_macro(cx) {
            return;
        }

        // If the item being implemented is not exported, then we don't need #[inline]
        if !cx.effective_visibilities.is_exported(impl_item.owner_id.def_id) {
            return;
        }

        let desc = match impl_item.kind {
            hir::ImplItemKind::Fn(..) => "a method",
            hir::ImplItemKind::Const(..) | hir::ImplItemKind::Type(_) => return,
        };

        let assoc_item = cx.tcx.associated_item(impl_item.owner_id);
        let container_id = assoc_item.container_id(cx.tcx);
        let trait_def_id = match assoc_item.container {
            AssocItemContainer::Trait => Some(container_id),
            AssocItemContainer::Impl => cx.tcx.impl_trait_ref(container_id).map(|t| t.skip_binder().def_id),
        };

        if let Some(trait_def_id) = trait_def_id
            && trait_def_id.is_local()
            && !cx.effective_visibilities.is_exported(impl_item.owner_id.def_id)
        {
            // If a trait is being implemented for an item, and the
            // trait is not exported, we don't need #[inline]
            return;
        }

        let attrs = cx.tcx.hir_attrs(impl_item.hir_id());
        check_missing_inline_attrs(cx, attrs, impl_item.span, desc);
    }
}
