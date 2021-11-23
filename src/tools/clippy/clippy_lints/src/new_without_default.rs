use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::return_ty;
use clippy_utils::source::snippet;
use clippy_utils::sugg::DiagnosticBuilderExt;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::HirIdSet;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::TyS;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for types with a `fn new() -> Self` method and no
    /// implementation of
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html).
    ///
    /// ### Why is this bad?
    /// The user might expect to be able to use
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html) as the
    /// type can be constructed without arguments.
    ///
    /// ### Example
    /// ```ignore
    /// struct Foo(Bar);
    ///
    /// impl Foo {
    ///     fn new() -> Self {
    ///         Foo(Bar::new())
    ///     }
    /// }
    /// ```
    ///
    /// To fix the lint, add a `Default` implementation that delegates to `new`:
    ///
    /// ```ignore
    /// struct Foo(Bar);
    ///
    /// impl Default for Foo {
    ///     fn default() -> Self {
    ///         Foo::new()
    ///     }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEW_WITHOUT_DEFAULT,
    style,
    "`fn new() -> Self` method without `Default` implementation"
}

#[derive(Clone, Default)]
pub struct NewWithoutDefault {
    impling_types: Option<HirIdSet>,
}

impl_lint_pass!(NewWithoutDefault => [NEW_WITHOUT_DEFAULT]);

impl<'tcx> LateLintPass<'tcx> for NewWithoutDefault {
    #[allow(clippy::too_many_lines)]
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if let hir::ItemKind::Impl(hir::Impl {
            of_trait: None,
            ref generics,
            self_ty: impl_self_ty,
            items,
            ..
        }) = item.kind
        {
            for assoc_item in items {
                if assoc_item.kind == (hir::AssocItemKind::Fn { has_self: false }) {
                    let impl_item = cx.tcx.hir().impl_item(assoc_item.id);
                    if in_external_macro(cx.sess(), impl_item.span) {
                        return;
                    }
                    if let hir::ImplItemKind::Fn(ref sig, _) = impl_item.kind {
                        let name = impl_item.ident.name;
                        let id = impl_item.hir_id();
                        if sig.header.constness == hir::Constness::Const {
                            // can't be implemented by default
                            return;
                        }
                        if sig.header.unsafety == hir::Unsafety::Unsafe {
                            // can't be implemented for unsafe new
                            return;
                        }
                        if impl_item
                            .generics
                            .params
                            .iter()
                            .any(|gen| matches!(gen.kind, hir::GenericParamKind::Type { .. }))
                        {
                            // when the result of `new()` depends on a type parameter we should not require
                            // an
                            // impl of `Default`
                            return;
                        }
                        if_chain! {
                            if sig.decl.inputs.is_empty();
                            if name == sym::new;
                            if cx.access_levels.is_reachable(impl_item.def_id);
                            let self_def_id = cx.tcx.hir().local_def_id(cx.tcx.hir().get_parent_item(id));
                            let self_ty = cx.tcx.type_of(self_def_id);
                            if TyS::same_type(self_ty, return_ty(cx, id));
                            if let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default);
                            then {
                                if self.impling_types.is_none() {
                                    let mut impls = HirIdSet::default();
                                    cx.tcx.for_each_impl(default_trait_id, |d| {
                                        if let Some(ty_def) = cx.tcx.type_of(d).ty_adt_def() {
                                            if let Some(local_def_id) = ty_def.did.as_local() {
                                                impls.insert(cx.tcx.hir().local_def_id_to_hir_id(local_def_id));
                                            }
                                        }
                                    });
                                    self.impling_types = Some(impls);
                                }

                                // Check if a Default implementation exists for the Self type, regardless of
                                // generics
                                if_chain! {
                                    if let Some(ref impling_types) = self.impling_types;
                                    if let Some(self_def) = cx.tcx.type_of(self_def_id).ty_adt_def();
                                    if let Some(self_local_did) = self_def.did.as_local();
                                    let self_id = cx.tcx.hir().local_def_id_to_hir_id(self_local_did);
                                    if impling_types.contains(&self_id);
                                    then {
                                        return;
                                    }
                                }

                                let generics_sugg = snippet(cx, generics.span, "");
                                let self_ty_fmt = self_ty.to_string();
                                let self_type_snip = snippet(cx, impl_self_ty.span, &self_ty_fmt);
                                span_lint_hir_and_then(
                                    cx,
                                    NEW_WITHOUT_DEFAULT,
                                    id,
                                    impl_item.span,
                                    &format!(
                                        "you should consider adding a `Default` implementation for `{}`",
                                        self_type_snip
                                    ),
                                    |diag| {
                                        diag.suggest_prepend_item(
                                            cx,
                                            item.span,
                                            "try adding this",
                                            &create_new_without_default_suggest_msg(&self_type_snip, &generics_sugg),
                                            Applicability::MaybeIncorrect,
                                        );
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn create_new_without_default_suggest_msg(self_type_snip: &str, generics_sugg: &str) -> String {
    #[rustfmt::skip]
    format!(
"impl{} Default for {} {{
    fn default() -> Self {{
        Self::new()
    }}
}}", generics_sugg, self_type_snip)
}
