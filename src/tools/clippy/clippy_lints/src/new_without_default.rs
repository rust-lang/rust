use crate::utils::paths;
use crate::utils::sugg::DiagnosticBuilderExt;
use crate::utils::{get_trait_def_id, return_ty, span_lint_hir_and_then};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::HirIdSet;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{Ty, TyS};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:** Checks for types with a `fn new() -> Self` method and no
    /// implementation of
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html).
    ///
    /// **Why is this bad?** The user might expect to be able to use
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html) as the
    /// type can be constructed without arguments.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
    ///
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
        if let hir::ItemKind::Impl {
            of_trait: None, items, ..
        } = item.kind
        {
            for assoc_item in items {
                if let hir::AssocItemKind::Fn { has_self: false } = assoc_item.kind {
                    let impl_item = cx.tcx.hir().impl_item(assoc_item.id);
                    if in_external_macro(cx.sess(), impl_item.span) {
                        return;
                    }
                    if let hir::ImplItemKind::Fn(ref sig, _) = impl_item.kind {
                        let name = impl_item.ident.name;
                        let id = impl_item.hir_id;
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
                        if sig.decl.inputs.is_empty() && name == sym::new && cx.access_levels.is_reachable(id) {
                            let self_def_id = cx.tcx.hir().local_def_id(cx.tcx.hir().get_parent_item(id));
                            let self_ty = cx.tcx.type_of(self_def_id);
                            if_chain! {
                                if TyS::same_type(self_ty, return_ty(cx, id));
                                if let Some(default_trait_id) = get_trait_def_id(cx, &paths::DEFAULT_TRAIT);
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
                                        then {
                                            let self_id = cx.tcx.hir().local_def_id_to_hir_id(self_local_did);
                                            if impling_types.contains(&self_id) {
                                                return;
                                            }
                                        }
                                    }

                                    span_lint_hir_and_then(
                                        cx,
                                        NEW_WITHOUT_DEFAULT,
                                        id,
                                        impl_item.span,
                                        &format!(
                                            "you should consider adding a `Default` implementation for `{}`",
                                            self_ty
                                        ),
                                        |diag| {
                                            diag.suggest_prepend_item(
                                                cx,
                                                item.span,
                                                "try this",
                                                &create_new_without_default_suggest_msg(self_ty),
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
}

fn create_new_without_default_suggest_msg(ty: Ty<'_>) -> String {
    #[rustfmt::skip]
    format!(
"impl Default for {} {{
    fn default() -> Self {{
        Self::new()
    }}
}}", ty)
}
