use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_lint_allowed;
use clippy_utils::source::snippet;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_ast::ImplPolarity;
use rustc_hir::def_id::DefId;
use rustc_hir::{FieldDef, Item, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, GenericArgKind, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns about a `Send` implementation for a type that
    /// contains fields that are not safe to be sent across threads.
    /// It tries to detect fields that can cause a soundness issue
    /// when sent to another thread (e.g., `Rc`) while allowing `!Send` fields
    /// that are expected to exist in a `Send` type, such as raw pointers.
    ///
    /// ### Why is this bad?
    /// Sending the struct to another thread effectively sends all of its fields,
    /// and the fields that do not implement `Send` can lead to soundness bugs
    /// such as data races when accessed in a thread
    /// that is different from the thread that created it.
    ///
    /// See:
    /// * [*The Rustonomicon* about *Send and Sync*](https://doc.rust-lang.org/nomicon/send-and-sync.html)
    /// * [The documentation of `Send`](https://doc.rust-lang.org/std/marker/trait.Send.html)
    ///
    /// ### Known Problems
    /// This lint relies on heuristics to distinguish types that are actually
    /// unsafe to be sent across threads and `!Send` types that are expected to
    /// exist in  `Send` type. Its rule can filter out basic cases such as
    /// `Vec<*const T>`, but it's not perfect. Feel free to create an issue if
    /// you have a suggestion on how this heuristic can be improved.
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
    /// or specify correct bounds on generic type parameters (`T: Send`).
    #[clippy::version = "1.57.0"]
    pub NON_SEND_FIELDS_IN_SEND_TY,
    nursery,
    "there is a field that is not safe to be sent to another thread in a `Send` struct"
}

pub struct NonSendFieldInSendTy {
    enable_raw_pointer_heuristic: bool,
}

impl NonSendFieldInSendTy {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            enable_raw_pointer_heuristic: conf.enable_raw_pointer_heuristic_for_send,
        }
    }
}

impl_lint_pass!(NonSendFieldInSendTy => [NON_SEND_FIELDS_IN_SEND_TY]);

impl<'tcx> LateLintPass<'tcx> for NonSendFieldInSendTy {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let ty_allowed_in_send = if self.enable_raw_pointer_heuristic {
            ty_allowed_with_raw_pointer_heuristic
        } else {
            ty_allowed_without_raw_pointer_heuristic
        };

        // Checks if we are in `Send` impl item.
        // We start from `Send` impl instead of `check_field_def()` because
        // single `AdtDef` may have multiple `Send` impls due to generic
        // parameters, and the lint is much easier to implement in this way.
        if !in_external_macro(cx.tcx.sess, item.span)
            && let Some(send_trait) = cx.tcx.get_diagnostic_item(sym::Send)
            && let ItemKind::Impl(hir_impl) = &item.kind
            && let Some(trait_ref) = &hir_impl.of_trait
            && let Some(trait_id) = trait_ref.trait_def_id()
            && send_trait == trait_id
            && hir_impl.polarity == ImplPolarity::Positive
            && let Some(ty_trait_ref) = cx.tcx.impl_trait_ref(item.owner_id)
            && let self_ty = ty_trait_ref.instantiate_identity().self_ty()
            && let ty::Adt(adt_def, impl_trait_args) = self_ty.kind()
        {
            let mut non_send_fields = Vec::new();

            for variant in adt_def.variants() {
                for field in &variant.fields {
                    if let Some(field_hir_id) = field
                        .did
                        .as_local()
                        .map(|local_def_id| cx.tcx.local_def_id_to_hir_id(local_def_id))
                        && !is_lint_allowed(cx, NON_SEND_FIELDS_IN_SEND_TY, field_hir_id)
                        && let field_ty = field.ty(cx.tcx, impl_trait_args)
                        && !ty_allowed_in_send(cx, field_ty, send_trait)
                        && let Node::Field(field_def) = cx.tcx.hir_node(field_hir_id)
                    {
                        non_send_fields.push(NonSendField {
                            def: field_def,
                            ty: field_ty,
                            generic_params: collect_generic_params(field_ty),
                        });
                    }
                }
            }

            if !non_send_fields.is_empty() {
                span_lint_and_then(
                    cx,
                    NON_SEND_FIELDS_IN_SEND_TY,
                    item.span,
                    format!(
                        "some fields in `{}` are not safe to be sent to another thread",
                        snippet(cx, hir_impl.self_ty.span, "Unknown")
                    ),
                    |diag| {
                        for field in non_send_fields {
                            diag.span_note(
                                field.def.span,
                                format!(
                                    "it is not safe to send field `{}` to another thread",
                                    field.def.ident.name
                                ),
                            );

                            match field.generic_params.len() {
                                0 => diag.help("use a thread-safe type that implements `Send`"),
                                1 if is_ty_param(field.ty) => {
                                    diag.help(format!("add `{}: Send` bound in `Send` impl", field.ty))
                                },
                                _ => diag.help(format!(
                                    "add bounds on type parameter{} `{}` that satisfy `{}: Send`",
                                    if field.generic_params.len() > 1 { "s" } else { "" },
                                    field.generic_params_string(),
                                    snippet(cx, field.def.ty.span, "Unknown"),
                                )),
                            };
                        }
                    },
                );
            }
        }
    }
}

struct NonSendField<'tcx> {
    def: &'tcx FieldDef<'tcx>,
    ty: Ty<'tcx>,
    generic_params: Vec<Ty<'tcx>>,
}

impl NonSendField<'_> {
    fn generic_params_string(&self) -> String {
        self.generic_params
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Given a type, collect all of its generic parameters.
/// Example: `MyStruct<P, Box<Q, R>>` => `vec![P, Q, R]`
fn collect_generic_params(ty: Ty<'_>) -> Vec<Ty<'_>> {
    ty.walk()
        .filter_map(|inner| match inner.unpack() {
            GenericArgKind::Type(inner_ty) => Some(inner_ty),
            _ => None,
        })
        .filter(|&inner_ty| is_ty_param(inner_ty))
        .collect()
}

/// Be more strict when the heuristic is disabled
fn ty_allowed_without_raw_pointer_heuristic<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, send_trait: DefId) -> bool {
    if implements_trait(cx, ty, send_trait, &[]) {
        return true;
    }

    if is_copy(cx, ty) && !contains_pointer_like(cx, ty) {
        return true;
    }

    false
}

/// Heuristic to allow cases like `Vec<*const u8>`
fn ty_allowed_with_raw_pointer_heuristic<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, send_trait: DefId) -> bool {
    if implements_trait(cx, ty, send_trait, &[]) || is_copy(cx, ty) {
        return true;
    }

    // The type is known to be `!Send` and `!Copy`
    match ty.kind() {
        ty::Tuple(fields) => fields
            .iter()
            .all(|ty| ty_allowed_with_raw_pointer_heuristic(cx, ty, send_trait)),
        ty::Array(ty, _) | ty::Slice(ty) => ty_allowed_with_raw_pointer_heuristic(cx, *ty, send_trait),
        ty::Adt(_, args) => {
            if contains_pointer_like(cx, ty) {
                // descends only if ADT contains any raw pointers
                args.iter().all(|generic_arg| match generic_arg.unpack() {
                    GenericArgKind::Type(ty) => ty_allowed_with_raw_pointer_heuristic(cx, ty, send_trait),
                    // Lifetimes and const generics are not solid part of ADT and ignored
                    GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => true,
                })
            } else {
                false
            }
        },
        // Raw pointers are `!Send` but allowed by the heuristic
        ty::RawPtr(_, _) => true,
        _ => false,
    }
}

/// Checks if the type contains any pointer-like types in args (including nested ones)
fn contains_pointer_like<'tcx>(cx: &LateContext<'tcx>, target_ty: Ty<'tcx>) -> bool {
    for ty_node in target_ty.walk() {
        if let GenericArgKind::Type(inner_ty) = ty_node.unpack() {
            match inner_ty.kind() {
                ty::RawPtr(_, _) => {
                    return true;
                },
                ty::Adt(adt_def, _) => {
                    if cx.tcx.is_diagnostic_item(sym::NonNull, adt_def.did()) {
                        return true;
                    }
                },
                _ => (),
            }
        }
    }

    false
}

/// Returns `true` if the type is a type parameter such as `T`.
fn is_ty_param(target_ty: Ty<'_>) -> bool {
    matches!(target_ty.kind(), ty::Param(_))
}
