use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast;
use rustc_hir as hir;
use rustc_lint::{self, LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// It lints if an exported function, method, trait method with default impl,
    /// or trait method impl is not `#[inline]`.
    ///
    /// ### Why is this bad?
    /// In general, it is not. Functions can be inlined across
    /// crates when that's profitable as long as any form of LTO is used. When LTO is disabled,
    /// functions that are not `#[inline]` cannot be inlined across crates. Certain types of crates
    /// might intend for most of the methods in their public API to be able to be inlined across
    /// crates even when LTO is disabled. For these types of crates, enabling this lint might make
    /// sense. It allows the crate to require all exported methods to be `#[inline]` by default, and
    /// then opt out for specific methods where this might not make sense.
    ///
    /// ### Example
    /// ```rust
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
    ///    fn private() {} // ok
    /// }
    ///
    /// impl Bar for Baz {
    ///   fn bar() {} // ok - Baz is not exported
    /// }
    ///
    /// pub struct PubBaz;
    /// impl PubBaz {
    ///    fn private() {} // ok
    ///    pub fn not_private() {} // missing #[inline]
    /// }
    ///
    /// impl Bar for PubBaz {
    ///    fn bar() {} // missing #[inline]
    ///    fn def_bar() {} // missing #[inline]
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MISSING_INLINE_IN_PUBLIC_ITEMS,
    restriction,
    "detects missing `#[inline]` attribute for public callables (functions, trait methods, methods...)"
}

fn check_missing_inline_attrs(cx: &LateContext<'_>, attrs: &[ast::Attribute], sp: Span, desc: &'static str) {
    let has_inline = attrs.iter().any(|a| a.has_name(sym::inline));
    if !has_inline {
        span_lint(
            cx,
            MISSING_INLINE_IN_PUBLIC_ITEMS,
            sp,
            &format!("missing `#[inline]` for {desc}"),
        );
    }
}

fn is_executable_or_proc_macro(cx: &LateContext<'_>) -> bool {
    use rustc_session::config::CrateType;

    cx.tcx
        .sess
        .crate_types()
        .iter()
        .any(|t: &CrateType| matches!(t, CrateType::Executable | CrateType::ProcMacro))
}

declare_lint_pass!(MissingInline => [MISSING_INLINE_IN_PUBLIC_ITEMS]);

impl<'tcx> LateLintPass<'tcx> for MissingInline {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::Item<'_>) {
        if rustc_middle::lint::in_external_macro(cx.sess(), it.span) || is_executable_or_proc_macro(cx) {
            return;
        }

        if !cx.effective_visibilities.is_exported(it.owner_id.def_id) {
            return;
        }
        match it.kind {
            hir::ItemKind::Fn(..) => {
                let desc = "a function";
                let attrs = cx.tcx.hir().attrs(it.hir_id());
                check_missing_inline_attrs(cx, attrs, it.span, desc);
            },
            hir::ItemKind::Trait(ref _is_auto, ref _unsafe, _generics, _bounds, trait_items) => {
                // note: we need to check if the trait is exported so we can't use
                // `LateLintPass::check_trait_item` here.
                for tit in trait_items {
                    let tit_ = cx.tcx.hir().trait_item(tit.id);
                    match tit_.kind {
                        hir::TraitItemKind::Const(..) | hir::TraitItemKind::Type(..) => {},
                        hir::TraitItemKind::Fn(..) => {
                            if cx.tcx.impl_defaultness(tit.id.owner_id).has_value() {
                                // trait method with default body needs inline in case
                                // an impl is not provided
                                let desc = "a default trait method";
                                let item = cx.tcx.hir().trait_item(tit.id);
                                let attrs = cx.tcx.hir().attrs(item.hir_id());
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
            | hir::ItemKind::GlobalAsm(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::OpaqueTy(..)
            | hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::Impl { .. }
            | hir::ItemKind::Use(..) => {},
        };
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx hir::ImplItem<'_>) {
        use rustc_middle::ty::{ImplContainer, TraitContainer};
        if rustc_middle::lint::in_external_macro(cx.sess(), impl_item.span) || is_executable_or_proc_macro(cx) {
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
            TraitContainer => Some(container_id),
            ImplContainer => cx.tcx.impl_trait_ref(container_id).map(|t| t.def_id),
        };

        if let Some(trait_def_id) = trait_def_id {
            if trait_def_id.is_local() && !cx.effective_visibilities.is_exported(impl_item.owner_id.def_id) {
                // If a trait is being implemented for an item, and the
                // trait is not exported, we don't need #[inline]
                return;
            }
        }

        let attrs = cx.tcx.hir().attrs(impl_item.hir_id());
        check_missing_inline_attrs(cx, attrs, impl_item.span, desc);
    }
}
