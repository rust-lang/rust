use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::return_ty;
use clippy_utils::source::snippet;
use clippy_utils::sugg::DiagExt;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::HirIdSet;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::AssocKind;
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public types with a `pub fn new() -> Self` method and no
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
    /// pub struct Foo(Bar);
    ///
    /// impl Foo {
    ///     pub fn new() -> Self {
    ///         Foo(Bar::new())
    ///     }
    /// }
    /// ```
    ///
    /// To fix the lint, add a `Default` implementation that delegates to `new`:
    ///
    /// ```ignore
    /// pub struct Foo(Bar);
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
    "`pub fn new() -> Self` method without `Default` implementation"
}

#[derive(Clone, Default)]
pub struct NewWithoutDefault {
    impling_types: Option<HirIdSet>,
}

impl_lint_pass!(NewWithoutDefault => [NEW_WITHOUT_DEFAULT]);

impl<'tcx> LateLintPass<'tcx> for NewWithoutDefault {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if let hir::ItemKind::Impl(hir::Impl {
            of_trait: None,
            generics,
            self_ty: impl_self_ty,
            ..
        }) = item.kind
        {
            for assoc_item in cx
                .tcx
                .associated_items(item.owner_id.def_id)
                .filter_by_name_unhygienic(sym::new)
            {
                if let AssocKind::Fn { has_self: false, .. } = assoc_item.kind {
                    let impl_item = cx
                        .tcx
                        .hir_node_by_def_id(assoc_item.def_id.expect_local())
                        .expect_impl_item();
                    if impl_item.span.in_external_macro(cx.sess().source_map()) {
                        return;
                    }
                    if let hir::ImplItemKind::Fn(ref sig, _) = impl_item.kind {
                        let id = impl_item.owner_id;
                        if sig.header.is_unsafe() {
                            // can't be implemented for unsafe new
                            return;
                        }
                        if cx.tcx.is_doc_hidden(impl_item.owner_id.def_id) {
                            // shouldn't be implemented when it is hidden in docs
                            return;
                        }
                        if !impl_item.generics.params.is_empty() {
                            // when the result of `new()` depends on a parameter we should not require
                            // an impl of `Default`
                            return;
                        }
                        if sig.decl.inputs.is_empty()
                            && cx.effective_visibilities.is_reachable(impl_item.owner_id.def_id)
                            && let self_ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
                            && self_ty == return_ty(cx, impl_item.owner_id)
                            && let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
                        {
                            if self.impling_types.is_none() {
                                let mut impls = HirIdSet::default();
                                for &d in cx.tcx.local_trait_impls(default_trait_id) {
                                    let ty = cx.tcx.type_of(d).instantiate_identity();
                                    if let Some(ty_def) = ty.ty_adt_def()
                                        && let Some(local_def_id) = ty_def.did().as_local()
                                    {
                                        impls.insert(cx.tcx.local_def_id_to_hir_id(local_def_id));
                                    }
                                }
                                self.impling_types = Some(impls);
                            }

                            // Check if a Default implementation exists for the Self type, regardless of
                            // generics
                            if let Some(ref impling_types) = self.impling_types
                                && let self_def = cx.tcx.type_of(item.owner_id).instantiate_identity()
                                && let Some(self_def) = self_def.ty_adt_def()
                                && let Some(self_local_did) = self_def.did().as_local()
                                && let self_id = cx.tcx.local_def_id_to_hir_id(self_local_did)
                                && impling_types.contains(&self_id)
                            {
                                return;
                            }

                            let generics_sugg = snippet(cx, generics.span, "");
                            let where_clause_sugg = if generics.has_where_clause_predicates {
                                format!("\n{}\n", snippet(cx, generics.where_clause_span, ""))
                            } else {
                                String::new()
                            };
                            let self_ty_fmt = self_ty.to_string();
                            let self_type_snip = snippet(cx, impl_self_ty.span, &self_ty_fmt);
                            span_lint_hir_and_then(
                                cx,
                                NEW_WITHOUT_DEFAULT,
                                id.into(),
                                impl_item.span,
                                format!("you should consider adding a `Default` implementation for `{self_type_snip}`"),
                                |diag| {
                                    diag.suggest_prepend_item(
                                        cx,
                                        item.span,
                                        "try adding this",
                                        &create_new_without_default_suggest_msg(
                                            &self_type_snip,
                                            &generics_sugg,
                                            &where_clause_sugg,
                                        ),
                                        Applicability::MachineApplicable,
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

fn create_new_without_default_suggest_msg(
    self_type_snip: &str,
    generics_sugg: &str,
    where_clause_sugg: &str,
) -> String {
    #[rustfmt::skip]
    format!(
"impl{generics_sugg} Default for {self_type_snip}{where_clause_sugg} {{
    fn default() -> Self {{
        Self::new()
    }}
}}")
}
