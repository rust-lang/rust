use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_note};
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_ast::ImplPolarity;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, subst::GenericArgKind, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Warns about a field in a `Send` struct that is neither `Send` nor `Copy`.
    ///
    /// ### Why is this bad?
    /// Sending the struct to another thread and drops it there will also drop
    /// the field in the new thread. This effectively changes the ownership of
    /// the field type to the new thread and creates a soundness issue by
    /// breaking breaks the non-`Send` invariant.
    ///
    /// ### Known Problems
    /// Data structures that contain raw pointers may cause false positives.
    /// They are sometimes safe to be sent across threads but do not implement
    /// the `Send` trait. This lint has a heuristic to filter out basic cases
    /// such as `Vec<*const T>`, but it's not perfect.
    ///
    /// ### Example
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// // There is no `RC: Send` bound here
    /// unsafe impl<RC, T: Send> Send for ArcGuard<RC, T> {}
    ///
    /// #[derive(Debug, Clone)]
    /// pub struct ArcGuard<RC, T> {
    ///     inner: T,
    ///     // Possibly drops `Arc<RC>` (and in turn `RC`) on a wrong thread
    ///     head: Arc<RC>
    /// }
    /// ```
    pub NON_SEND_FIELD_IN_SEND_TY,
    nursery,
    "a field in a `Send` struct does not implement `Send`"
}

declare_lint_pass!(NonSendFieldInSendTy => [NON_SEND_FIELD_IN_SEND_TY]);

impl<'tcx> LateLintPass<'tcx> for NonSendFieldInSendTy {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let send_trait = cx.tcx.get_diagnostic_item(sym::send_trait).unwrap();

        // Check if we are in `Send` impl item
        if_chain! {
            if let ItemKind::Impl(hir_impl) = &item.kind;
            if let Some(trait_ref) = &hir_impl.of_trait;
            if let Some(trait_id) = trait_ref.trait_def_id();
            if send_trait == trait_id;
            if let ImplPolarity::Positive = hir_impl.polarity;
            if let Some(ty_trait_ref) = cx.tcx.impl_trait_ref(item.def_id);
            if let self_ty = ty_trait_ref.self_ty();
            if let ty::Adt(adt_def, impl_trait_substs) = self_ty.kind();
            then {
                for variant in &adt_def.variants {
                    for field in &variant.fields {
                        let field_ty = field.ty(cx.tcx, impl_trait_substs);

                        // TODO: substs rebase_onto

                        if raw_pointer_in_ty_def(cx, field_ty)
                            || implements_trait(cx, field_ty, send_trait, &[])
                            || is_copy(cx, field_ty)
                        {
                            continue;
                        }

                        if let Some(field_span) = cx.tcx.hir().span_if_local(field.did) {
                            if is_ty_param(field_ty) {
                                span_lint_and_help(
                                    cx,
                                    NON_SEND_FIELD_IN_SEND_TY,
                                    field_span,
                                    "a field in a `Send` struct does not implement `Send`",
                                    Some(item.span),
                                    &format!("add `{}: Send` in `Send` impl for `{}`", field_ty, self_ty),
                                )
                            } else {
                                span_lint_and_note(
                                    cx,
                                    NON_SEND_FIELD_IN_SEND_TY,
                                    field_span,
                                    "a field in a `Send` struct does not implement `Send`",
                                    Some(item.span),
                                    &format!(
                                        "type `{}` doesn't implement `Send` when `{}` is `Send`",
                                        field_ty, self_ty
                                    ),
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Returns `true` if the type itself is a raw pointer or has a raw pointer as a
/// generic parameter, e.g., `Vec<*const u8>`.
/// Note that it does not look into enum variants or struct fields.
fn raw_pointer_in_ty_def<'tcx>(cx: &LateContext<'tcx>, target_ty: Ty<'tcx>) -> bool {
    for ty_node in target_ty.walk(cx.tcx) {
        if_chain! {
            if let GenericArgKind::Type(inner_ty) = ty_node.unpack();
            if let ty::RawPtr(_) = inner_ty.kind();
            then {
                return true;
            }
        }
    }

    false
}

/// Returns `true` if the type is a type parameter such as `T`.
fn is_ty_param<'tcx>(target_ty: Ty<'tcx>) -> bool {
    matches!(target_ty.kind(), ty::Param(_))
}
