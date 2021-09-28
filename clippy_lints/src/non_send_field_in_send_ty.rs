use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_lint_allowed;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_ast::ImplPolarity;
use rustc_hir::def_id::DefId;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, subst::GenericArgKind, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Symbol;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Warns about a field in a `Send` struct that is neither `Send` nor `Copy`.
    ///
    /// ### Why is this bad?
    /// Sending the struct to another thread will transfer the ownership to
    /// the new thread by dropping in the current thread during the transfer.
    /// This causes soundness issues for non-`Send` fields, as they are also
    /// dropped and might not be set up to handle this.
    ///
    /// See:
    /// * [*The Rustonomicon* about *Send and Sync*](https://doc.rust-lang.org/nomicon/send-and-sync.html)
    /// * [The documentation of `Send`](https://doc.rust-lang.org/std/marker/trait.Send.html)
    ///
    /// ### Known Problems
    /// Data structures that contain raw pointers may cause false positives.
    /// They are sometimes safe to be sent across threads but do not implement
    /// the `Send` trait. This lint has a heuristic to filter out basic cases
    /// such as `Vec<*const T>`, but it's not perfect. Feel free to create an
    /// issue if you have a suggestion on how this heuristic can be improved.
    ///
    /// ### Example
    /// ```rust,ignore
    /// struct ExampleStruct<T> {
    ///     rc_is_not_send: Rc<String>,
    ///     unbounded_generic_field: T,
    /// }
    ///
    /// // This impl is unsound because it allows sending `!Send` types through `ExampleStruct`
    /// unsafe impl<T> Send for ExampleStruct<T> {}
    /// ```
    /// Use thread-safe types like [`std::sync::Arc`](https://doc.rust-lang.org/std/sync/struct.Arc.html)
    /// and specify correct bounds on generic type parameters (`T: Send`).
    pub NON_SEND_FIELD_IN_SEND_TY,
    nursery,
    "there is field that does not implement `Send` in a `Send` struct"
}

declare_lint_pass!(NonSendFieldInSendTy => [NON_SEND_FIELD_IN_SEND_TY]);

impl<'tcx> LateLintPass<'tcx> for NonSendFieldInSendTy {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        // Checks if we are in `Send` impl item.
        // We start from `Send` impl instead of `check_field_def()` because
        // single `AdtDef` may have multiple `Send` impls due to generic
        // parameters, and the lint is much easier to implement in this way.
        if_chain! {
            if let Some(send_trait) = cx.tcx.get_diagnostic_item(sym::send_trait);
            if let ItemKind::Impl(hir_impl) = &item.kind;
            if let Some(trait_ref) = &hir_impl.of_trait;
            if let Some(trait_id) = trait_ref.trait_def_id();
            if send_trait == trait_id;
            if let ImplPolarity::Positive = hir_impl.polarity;
            if let Some(ty_trait_ref) = cx.tcx.impl_trait_ref(item.def_id);
            if let self_ty = ty_trait_ref.self_ty();
            if let ty::Adt(adt_def, impl_trait_substs) = self_ty.kind();
            then {
                let mut non_send_fields = Vec::new();

                let hir_map = cx.tcx.hir();
                for variant in &adt_def.variants {
                    for field in &variant.fields {
                        if_chain! {
                            if let Some(field_hir_id) = field
                                .did
                                .as_local()
                                .map(|local_def_id| hir_map.local_def_id_to_hir_id(local_def_id));
                            if !is_lint_allowed(cx, NON_SEND_FIELD_IN_SEND_TY, field_hir_id);
                            if let field_ty = field.ty(cx.tcx, impl_trait_substs);
                            if !ty_allowed_in_send(cx, field_ty, send_trait);
                            if let Some(field_span) = hir_map.span_if_local(field.did);
                            then {
                                non_send_fields.push(NonSendField {
                                    name: hir_map.name(field_hir_id),
                                    span: field_span,
                                    ty: field_ty,
                                    generic_params: collect_generic_params(cx, field_ty),
                                })
                            }
                        }
                    }
                }

                if !non_send_fields.is_empty() {
                    span_lint_and_then(
                        cx,
                        NON_SEND_FIELD_IN_SEND_TY,
                        item.span,
                        &format!(
                            "this implementation is unsound, as some fields in `{}` are `!Send`",
                            self_ty
                        ),
                        |diag| {
                            for field in non_send_fields {
                                diag.span_note(
                                    field.span,
                                    &format!("the field `{}` has type `{}` which is not `Send`", field.name, field.ty),
                                );

                                match field.generic_params.len() {
                                    0 => diag.help("use a thread-safe type that implements `Send`"),
                                    1 if is_ty_param(field.ty) => diag.help(&format!("add `{}: Send` bound in `Send` impl", field.ty)),
                                    _ => diag.help(&format!(
                                        "add bounds on type parameter{} `{}` that satisfy `{}: Send`",
                                        if field.generic_params.len() > 1 { "s" } else { "" },
                                        field.generic_params_string(),
                                        field.ty
                                    )),
                                };
                            }
                        },
                    )
                }
            }
        }
    }
}

struct NonSendField<'tcx> {
    name: Symbol,
    span: Span,
    ty: Ty<'tcx>,
    generic_params: Vec<Ty<'tcx>>,
}

impl<'tcx> NonSendField<'tcx> {
    fn generic_params_string(&self) -> String {
        self.generic_params
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn collect_generic_params<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Vec<Ty<'tcx>> {
    ty.walk(cx.tcx)
        .filter_map(|inner| match inner.unpack() {
            GenericArgKind::Type(inner_ty) => Some(inner_ty),
            _ => None,
        })
        .filter(|&inner_ty| is_ty_param(inner_ty))
        .collect()
}

fn ty_allowed_in_send<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, send_trait: DefId) -> bool {
    raw_pointer_in_ty_def(cx, ty) || implements_trait(cx, ty, send_trait, &[]) || is_copy(cx, ty)
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
fn is_ty_param(target_ty: Ty<'_>) -> bool {
    matches!(target_ty.kind(), ty::Param(_))
}
