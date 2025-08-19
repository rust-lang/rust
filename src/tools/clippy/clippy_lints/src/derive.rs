use std::ops::ControlFlow;

use clippy_utils::diagnostics::{span_lint_and_note, span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::ty::{implements_trait, implements_trait_with_env, is_copy};
use clippy_utils::{has_non_exhaustive_attr, is_lint_allowed, paths};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{FnKind, Visitor, walk_expr, walk_fn, walk_item};
use rustc_hir::{self as hir, BlockCheckMode, BodyId, Expr, ExprKind, FnDecl, Impl, Item, ItemKind, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{
    self, ClauseKind, GenericArgKind, GenericParamDefKind, ParamEnv, TraitPredicate, Ty, TyCtxt, Upcast,
};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Lints against manual `PartialEq` implementations for types with a derived `Hash`
    /// implementation.
    ///
    /// ### Why is this bad?
    /// The implementation of these traits must agree (for
    /// example for use with `HashMap`) so it’s probably a bad idea to use a
    /// default-generated `Hash` implementation with an explicitly defined
    /// `PartialEq`. In particular, the following must hold for any type:
    ///
    /// ```text
    /// k1 == k2 ⇒ hash(k1) == hash(k2)
    /// ```
    ///
    /// ### Example
    /// ```ignore
    /// #[derive(Hash)]
    /// struct Foo;
    ///
    /// impl PartialEq for Foo {
    ///     ...
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DERIVED_HASH_WITH_MANUAL_EQ,
    correctness,
    "deriving `Hash` but implementing `PartialEq` explicitly"
}

declare_clippy_lint! {
    /// ### What it does
    /// Lints against manual `PartialOrd` and `Ord` implementations for types with a derived `Ord`
    /// or `PartialOrd` implementation.
    ///
    /// ### Why is this bad?
    /// The implementation of these traits must agree (for
    /// example for use with `sort`) so it’s probably a bad idea to use a
    /// default-generated `Ord` implementation with an explicitly defined
    /// `PartialOrd`. In particular, the following must hold for any type
    /// implementing `Ord`:
    ///
    /// ```text
    /// k1.cmp(&k2) == k1.partial_cmp(&k2).unwrap()
    /// ```
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[derive(Ord, PartialEq, Eq)]
    /// struct Foo;
    ///
    /// impl PartialOrd for Foo {
    ///     ...
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// #[derive(PartialEq, Eq)]
    /// struct Foo;
    ///
    /// impl PartialOrd for Foo {
    ///     fn partial_cmp(&self, other: &Foo) -> Option<Ordering> {
    ///        Some(self.cmp(other))
    ///     }
    /// }
    ///
    /// impl Ord for Foo {
    ///     ...
    /// }
    /// ```
    /// or, if you don't need a custom ordering:
    /// ```rust,ignore
    /// #[derive(Ord, PartialOrd, PartialEq, Eq)]
    /// struct Foo;
    /// ```
    #[clippy::version = "1.47.0"]
    pub DERIVE_ORD_XOR_PARTIAL_ORD,
    correctness,
    "deriving `Ord` but implementing `PartialOrd` explicitly"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for explicit `Clone` implementations for `Copy`
    /// types.
    ///
    /// ### Why is this bad?
    /// To avoid surprising behavior, these traits should
    /// agree and the behavior of `Copy` cannot be overridden. In almost all
    /// situations a `Copy` type should have a `Clone` implementation that does
    /// nothing more than copy the object, which is what `#[derive(Copy, Clone)]`
    /// gets you.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[derive(Copy)]
    /// struct Foo;
    ///
    /// impl Clone for Foo {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EXPL_IMPL_CLONE_ON_COPY,
    pedantic,
    "implementing `Clone` explicitly on `Copy` types"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for deriving `serde::Deserialize` on a type that
    /// has methods using `unsafe`.
    ///
    /// ### Why is this bad?
    /// Deriving `serde::Deserialize` will create a constructor
    /// that may violate invariants held by another constructor.
    ///
    /// ### Example
    /// ```rust,ignore
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// pub struct Foo {
    ///     // ..
    /// }
    ///
    /// impl Foo {
    ///     pub fn new() -> Self {
    ///         // setup here ..
    ///     }
    ///
    ///     pub unsafe fn parts() -> (&str, &str) {
    ///         // assumes invariants hold
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.45.0"]
    pub UNSAFE_DERIVE_DESERIALIZE,
    pedantic,
    "deriving `serde::Deserialize` on a type that has methods using `unsafe`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for types that derive `PartialEq` and could implement `Eq`.
    ///
    /// ### Why is this bad?
    /// If a type `T` derives `PartialEq` and all of its members implement `Eq`,
    /// then `T` can always implement `Eq`. Implementing `Eq` allows `T` to be used
    /// in APIs that require `Eq` types. It also allows structs containing `T` to derive
    /// `Eq` themselves.
    ///
    /// ### Example
    /// ```no_run
    /// #[derive(PartialEq)]
    /// struct Foo {
    ///     i_am_eq: i32,
    ///     i_am_eq_too: Vec<String>,
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[derive(PartialEq, Eq)]
    /// struct Foo {
    ///     i_am_eq: i32,
    ///     i_am_eq_too: Vec<String>,
    /// }
    /// ```
    #[clippy::version = "1.63.0"]
    pub DERIVE_PARTIAL_EQ_WITHOUT_EQ,
    nursery,
    "deriving `PartialEq` on a type that can implement `Eq`, without implementing `Eq`"
}

declare_lint_pass!(Derive => [
    EXPL_IMPL_CLONE_ON_COPY,
    DERIVED_HASH_WITH_MANUAL_EQ,
    DERIVE_ORD_XOR_PARTIAL_ORD,
    UNSAFE_DERIVE_DESERIALIZE,
    DERIVE_PARTIAL_EQ_WITHOUT_EQ
]);

impl<'tcx> LateLintPass<'tcx> for Derive {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Impl(Impl {
            of_trait: Some(of_trait),
            ..
        }) = item.kind
        {
            let trait_ref = &of_trait.trait_ref;
            let ty = cx.tcx.type_of(item.owner_id).instantiate_identity();
            let is_automatically_derived = cx.tcx.is_automatically_derived(item.owner_id.to_def_id());

            check_hash_peq(cx, item.span, trait_ref, ty, is_automatically_derived);
            check_ord_partial_ord(cx, item.span, trait_ref, ty, is_automatically_derived);

            if is_automatically_derived {
                check_unsafe_derive_deserialize(cx, item, trait_ref, ty);
                check_partial_eq_without_eq(cx, item.span, trait_ref, ty);
            } else {
                check_copy_clone(cx, item, trait_ref, ty);
            }
        }
    }
}

/// Implementation of the `DERIVED_HASH_WITH_MANUAL_EQ` lint.
fn check_hash_peq<'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    trait_ref: &hir::TraitRef<'_>,
    ty: Ty<'tcx>,
    hash_is_automatically_derived: bool,
) {
    if let Some(peq_trait_def_id) = cx.tcx.lang_items().eq_trait()
        && let Some(def_id) = trait_ref.trait_def_id()
        && cx.tcx.is_diagnostic_item(sym::Hash, def_id)
    {
        // Look for the PartialEq implementations for `ty`
        cx.tcx.for_each_relevant_impl(peq_trait_def_id, ty, |impl_id| {
            let peq_is_automatically_derived = cx.tcx.is_automatically_derived(impl_id);

            if !hash_is_automatically_derived || peq_is_automatically_derived {
                return;
            }

            let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

            // Only care about `impl PartialEq<Foo> for Foo`
            // For `impl PartialEq<B> for A, input_types is [A, B]
            if trait_ref.instantiate_identity().args.type_at(1) == ty {
                span_lint_and_then(
                    cx,
                    DERIVED_HASH_WITH_MANUAL_EQ,
                    span,
                    "you are deriving `Hash` but have implemented `PartialEq` explicitly",
                    |diag| {
                        if let Some(local_def_id) = impl_id.as_local() {
                            let hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id);
                            diag.span_note(cx.tcx.hir_span(hir_id), "`PartialEq` implemented here");
                        }
                    },
                );
            }
        });
    }
}

/// Implementation of the `DERIVE_ORD_XOR_PARTIAL_ORD` lint.
fn check_ord_partial_ord<'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    trait_ref: &hir::TraitRef<'_>,
    ty: Ty<'tcx>,
    ord_is_automatically_derived: bool,
) {
    if let Some(ord_trait_def_id) = cx.tcx.get_diagnostic_item(sym::Ord)
        && let Some(partial_ord_trait_def_id) = cx.tcx.lang_items().partial_ord_trait()
        && let Some(def_id) = &trait_ref.trait_def_id()
        && *def_id == ord_trait_def_id
    {
        // Look for the PartialOrd implementations for `ty`
        cx.tcx.for_each_relevant_impl(partial_ord_trait_def_id, ty, |impl_id| {
            let partial_ord_is_automatically_derived = cx.tcx.is_automatically_derived(impl_id);

            if partial_ord_is_automatically_derived == ord_is_automatically_derived {
                return;
            }

            let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

            // Only care about `impl PartialOrd<Foo> for Foo`
            // For `impl PartialOrd<B> for A, input_types is [A, B]
            if trait_ref.instantiate_identity().args.type_at(1) == ty {
                let mess = if partial_ord_is_automatically_derived {
                    "you are implementing `Ord` explicitly but have derived `PartialOrd`"
                } else {
                    "you are deriving `Ord` but have implemented `PartialOrd` explicitly"
                };

                span_lint_and_then(cx, DERIVE_ORD_XOR_PARTIAL_ORD, span, mess, |diag| {
                    if let Some(local_def_id) = impl_id.as_local() {
                        let hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id);
                        diag.span_note(cx.tcx.hir_span(hir_id), "`PartialOrd` implemented here");
                    }
                });
            }
        });
    }
}

/// Implementation of the `EXPL_IMPL_CLONE_ON_COPY` lint.
fn check_copy_clone<'tcx>(cx: &LateContext<'tcx>, item: &Item<'_>, trait_ref: &hir::TraitRef<'_>, ty: Ty<'tcx>) {
    let clone_id = match cx.tcx.lang_items().clone_trait() {
        Some(id) if trait_ref.trait_def_id() == Some(id) => id,
        _ => return,
    };
    let Some(copy_id) = cx.tcx.lang_items().copy_trait() else {
        return;
    };
    let (ty_adt, ty_subs) = match *ty.kind() {
        // Unions can't derive clone.
        ty::Adt(adt, subs) if !adt.is_union() => (adt, subs),
        _ => return,
    };
    // If the current self type doesn't implement Copy (due to generic constraints), search to see if
    // there's a Copy impl for any instance of the adt.
    if !is_copy(cx, ty) {
        if ty_subs.non_erasable_generics().next().is_some() {
            let has_copy_impl = cx.tcx.local_trait_impls(copy_id).iter().any(|&id| {
                matches!(cx.tcx.type_of(id).instantiate_identity().kind(), ty::Adt(adt, _)
                                        if ty_adt.did() == adt.did())
            });
            if !has_copy_impl {
                return;
            }
        } else {
            return;
        }
    }
    // Derive constrains all generic types to requiring Clone. Check if any type is not constrained for
    // this impl.
    if ty_subs.types().any(|ty| !implements_trait(cx, ty, clone_id, &[])) {
        return;
    }
    // `#[repr(packed)]` structs with type/const parameters can't derive `Clone`.
    // https://github.com/rust-lang/rust-clippy/issues/10188
    if ty_adt.repr().packed()
        && ty_subs
            .iter()
            .any(|arg| matches!(arg.kind(), GenericArgKind::Type(_) | GenericArgKind::Const(_)))
    {
        return;
    }
    // The presence of `unsafe` fields prevents deriving `Clone` automatically
    if ty_adt.all_fields().any(|f| f.safety.is_unsafe()) {
        return;
    }

    span_lint_and_note(
        cx,
        EXPL_IMPL_CLONE_ON_COPY,
        item.span,
        "you are implementing `Clone` explicitly on a `Copy` type",
        Some(item.span),
        "consider deriving `Clone` or removing `Copy`",
    );
}

/// Implementation of the `UNSAFE_DERIVE_DESERIALIZE` lint.
fn check_unsafe_derive_deserialize<'tcx>(
    cx: &LateContext<'tcx>,
    item: &Item<'_>,
    trait_ref: &hir::TraitRef<'_>,
    ty: Ty<'tcx>,
) {
    fn has_unsafe<'tcx>(cx: &LateContext<'tcx>, item: &'tcx Item<'_>) -> bool {
        let mut visitor = UnsafeVisitor { cx };
        walk_item(&mut visitor, item).is_break()
    }

    if let Some(trait_def_id) = trait_ref.trait_def_id()
        && paths::SERDE_DESERIALIZE.matches(cx, trait_def_id)
        && let ty::Adt(def, _) = ty.kind()
        && let Some(local_def_id) = def.did().as_local()
        && let adt_hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id)
        && !is_lint_allowed(cx, UNSAFE_DERIVE_DESERIALIZE, adt_hir_id)
        && cx
            .tcx
            .inherent_impls(def.did())
            .iter()
            .map(|imp_did| cx.tcx.hir_expect_item(imp_did.expect_local()))
            .any(|imp| has_unsafe(cx, imp))
    {
        span_lint_hir_and_then(
            cx,
            UNSAFE_DERIVE_DESERIALIZE,
            adt_hir_id,
            item.span,
            "you are deriving `serde::Deserialize` on a type that has methods using `unsafe`",
            |diag| {
                diag.help(
                    "consider implementing `serde::Deserialize` manually. See https://serde.rs/impl-deserialize.html",
                );
            },
        );
    }
}

struct UnsafeVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for UnsafeVisitor<'_, 'tcx> {
    type Result = ControlFlow<()>;
    type NestedFilter = nested_filter::All;

    fn visit_fn(
        &mut self,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body_id: BodyId,
        _: Span,
        id: LocalDefId,
    ) -> Self::Result {
        if let Some(header) = kind.header()
            && header.is_unsafe()
        {
            ControlFlow::Break(())
        } else {
            walk_fn(self, kind, decl, body_id, id)
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) -> Self::Result {
        if let ExprKind::Block(block, _) = expr.kind
            && block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
            && block
                .span
                .source_callee()
                .and_then(|expr| expr.macro_def_id)
                .is_none_or(|did| !self.cx.tcx.is_diagnostic_item(sym::pin_macro, did))
        {
            return ControlFlow::Break(());
        }

        walk_expr(self, expr)
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

/// Implementation of the `DERIVE_PARTIAL_EQ_WITHOUT_EQ` lint.
fn check_partial_eq_without_eq<'tcx>(cx: &LateContext<'tcx>, span: Span, trait_ref: &hir::TraitRef<'_>, ty: Ty<'tcx>) {
    if let ty::Adt(adt, args) = ty.kind()
        && cx.tcx.visibility(adt.did()).is_public()
        && let Some(eq_trait_def_id) = cx.tcx.get_diagnostic_item(sym::Eq)
        && let Some(def_id) = trait_ref.trait_def_id()
        && cx.tcx.is_diagnostic_item(sym::PartialEq, def_id)
        && !has_non_exhaustive_attr(cx.tcx, *adt)
        && !ty_implements_eq_trait(cx.tcx, ty, eq_trait_def_id)
        && let typing_env = typing_env_for_derived_eq(cx.tcx, adt.did(), eq_trait_def_id)
        && let Some(local_def_id) = adt.did().as_local()
        // If all of our fields implement `Eq`, we can implement `Eq` too
        && adt
            .all_fields()
            .map(|f| f.ty(cx.tcx, args))
            .all(|ty| implements_trait_with_env(cx.tcx, typing_env, ty, eq_trait_def_id, None, &[]))
    {
        span_lint_hir_and_then(
            cx,
            DERIVE_PARTIAL_EQ_WITHOUT_EQ,
            cx.tcx.local_def_id_to_hir_id(local_def_id),
            span.ctxt().outer_expn_data().call_site,
            "you are deriving `PartialEq` and can implement `Eq`",
            |diag| {
                diag.span_suggestion(
                    span.ctxt().outer_expn_data().call_site,
                    "consider deriving `Eq` as well",
                    "PartialEq, Eq",
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

fn ty_implements_eq_trait<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, eq_trait_id: DefId) -> bool {
    tcx.non_blanket_impls_for_ty(eq_trait_id, ty).next().is_some()
}

/// Creates the `ParamEnv` used for the given type's derived `Eq` impl.
fn typing_env_for_derived_eq(tcx: TyCtxt<'_>, did: DefId, eq_trait_id: DefId) -> ty::TypingEnv<'_> {
    // Initial map from generic index to param def.
    // Vec<(param_def, needs_eq)>
    let mut params = tcx
        .generics_of(did)
        .own_params
        .iter()
        .map(|p| (p, matches!(p.kind, GenericParamDefKind::Type { .. })))
        .collect::<Vec<_>>();

    let ty_predicates = tcx.predicates_of(did).predicates;
    for (p, _) in ty_predicates {
        if let ClauseKind::Trait(p) = p.kind().skip_binder()
            && p.trait_ref.def_id == eq_trait_id
            && let ty::Param(self_ty) = p.trait_ref.self_ty().kind()
        {
            // Flag types which already have an `Eq` bound.
            params[self_ty.index as usize].1 = false;
        }
    }

    let param_env = ParamEnv::new(tcx.mk_clauses_from_iter(ty_predicates.iter().map(|&(p, _)| p).chain(
        params.iter().filter(|&&(_, needs_eq)| needs_eq).map(|&(param, _)| {
            ClauseKind::Trait(TraitPredicate {
                trait_ref: ty::TraitRef::new(tcx, eq_trait_id, [tcx.mk_param_from_def(param)]),
                polarity: ty::PredicatePolarity::Positive,
            })
            .upcast(tcx)
        }),
    )));
    ty::TypingEnv {
        typing_mode: ty::TypingMode::non_body_analysis(),
        param_env,
    }
}
