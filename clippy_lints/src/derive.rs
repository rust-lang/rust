use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_note, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::paths;
use clippy_utils::ty::{implements_trait, implements_trait_with_env, is_copy};
use clippy_utils::{is_lint_allowed, match_def_path};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{walk_expr, walk_fn, walk_item, FnKind, Visitor};
use rustc_hir::{
    self as hir, BlockCheckMode, BodyId, Constness, Expr, ExprKind, FnDecl, Impl, Item, ItemKind, UnsafeSource,
    Unsafety,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::traits::Reveal;
use rustc_middle::ty::{
    self, Binder, BoundConstness, Clause, GenericArgKind, GenericParamDefKind, ImplPolarity, ParamEnv, PredicateKind,
    TraitPredicate, Ty, TyCtxt,
};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::def_id::LocalDefId;
use rustc_span::source_map::Span;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for deriving `Hash` but implementing `PartialEq`
    /// explicitly or vice versa.
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
    /// Checks for deriving `Ord` but implementing `PartialOrd`
    /// explicitly or vice versa.
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
    /// that may violate invariants hold by another constructor.
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
    /// ```rust
    /// #[derive(PartialEq)]
    /// struct Foo {
    ///     i_am_eq: i32,
    ///     i_am_eq_too: Vec<String>,
    /// }
    /// ```
    /// Use instead:
    /// ```rust
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
            of_trait: Some(ref trait_ref),
            ..
        }) = item.kind
        {
            let ty = cx.tcx.bound_type_of(item.owner_id).subst_identity();
            let is_automatically_derived = cx.tcx.has_attr(item.owner_id.to_def_id(), sym::automatically_derived);

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
    if_chain! {
        if let Some(peq_trait_def_id) = cx.tcx.lang_items().eq_trait();
        if let Some(def_id) = trait_ref.trait_def_id();
        if cx.tcx.is_diagnostic_item(sym::Hash, def_id);
        then {
            // Look for the PartialEq implementations for `ty`
            cx.tcx.for_each_relevant_impl(peq_trait_def_id, ty, |impl_id| {
                let peq_is_automatically_derived = cx.tcx.has_attr(impl_id, sym::automatically_derived);

                if !hash_is_automatically_derived || peq_is_automatically_derived {
                    return;
                }

                let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

                // Only care about `impl PartialEq<Foo> for Foo`
                // For `impl PartialEq<B> for A, input_types is [A, B]
                if trait_ref.subst_identity().substs.type_at(1) == ty {
                    span_lint_and_then(
                        cx,
                        DERIVED_HASH_WITH_MANUAL_EQ,
                        span,
                        "you are deriving `Hash` but have implemented `PartialEq` explicitly",
                        |diag| {
                            if let Some(local_def_id) = impl_id.as_local() {
                                let hir_id = cx.tcx.hir().local_def_id_to_hir_id(local_def_id);
                                diag.span_note(
                                    cx.tcx.hir().span(hir_id),
                                    "`PartialEq` implemented here"
                                );
                            }
                        }
                    );
                }
            });
        }
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
    if_chain! {
        if let Some(ord_trait_def_id) = cx.tcx.get_diagnostic_item(sym::Ord);
        if let Some(partial_ord_trait_def_id) = cx.tcx.lang_items().partial_ord_trait();
        if let Some(def_id) = &trait_ref.trait_def_id();
        if *def_id == ord_trait_def_id;
        then {
            // Look for the PartialOrd implementations for `ty`
            cx.tcx.for_each_relevant_impl(partial_ord_trait_def_id, ty, |impl_id| {
                let partial_ord_is_automatically_derived = cx.tcx.has_attr(impl_id, sym::automatically_derived);

                if partial_ord_is_automatically_derived == ord_is_automatically_derived {
                    return;
                }

                let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

                // Only care about `impl PartialOrd<Foo> for Foo`
                // For `impl PartialOrd<B> for A, input_types is [A, B]
                if trait_ref.subst_identity().substs.type_at(1) == ty {
                    let mess = if partial_ord_is_automatically_derived {
                        "you are implementing `Ord` explicitly but have derived `PartialOrd`"
                    } else {
                        "you are deriving `Ord` but have implemented `PartialOrd` explicitly"
                    };

                    span_lint_and_then(
                        cx,
                        DERIVE_ORD_XOR_PARTIAL_ORD,
                        span,
                        mess,
                        |diag| {
                            if let Some(local_def_id) = impl_id.as_local() {
                                let hir_id = cx.tcx.hir().local_def_id_to_hir_id(local_def_id);
                                diag.span_note(
                                    cx.tcx.hir().span(hir_id),
                                    "`PartialOrd` implemented here"
                                );
                            }
                        }
                    );
                }
            });
        }
    }
}

/// Implementation of the `EXPL_IMPL_CLONE_ON_COPY` lint.
fn check_copy_clone<'tcx>(cx: &LateContext<'tcx>, item: &Item<'_>, trait_ref: &hir::TraitRef<'_>, ty: Ty<'tcx>) {
    let clone_id = match cx.tcx.lang_items().clone_trait() {
        Some(id) if trait_ref.trait_def_id() == Some(id) => id,
        _ => return,
    };
    let Some(copy_id) = cx.tcx.lang_items().copy_trait() else { return };
    let (ty_adt, ty_subs) = match *ty.kind() {
        // Unions can't derive clone.
        ty::Adt(adt, subs) if !adt.is_union() => (adt, subs),
        _ => return,
    };
    // If the current self type doesn't implement Copy (due to generic constraints), search to see if
    // there's a Copy impl for any instance of the adt.
    if !is_copy(cx, ty) {
        if ty_subs.non_erasable_generics().next().is_some() {
            let has_copy_impl = cx.tcx.all_local_trait_impls(()).get(&copy_id).map_or(false, |impls| {
                impls
                    .iter()
                    .any(|&id| matches!(cx.tcx.bound_type_of(id).subst_identity().kind(), ty::Adt(adt, _) if ty_adt.did() == adt.did()))
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
            .any(|arg| matches!(arg.unpack(), GenericArgKind::Type(_) | GenericArgKind::Const(_)))
    {
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
        let mut visitor = UnsafeVisitor { cx, has_unsafe: false };
        walk_item(&mut visitor, item);
        visitor.has_unsafe
    }

    if_chain! {
        if let Some(trait_def_id) = trait_ref.trait_def_id();
        if match_def_path(cx, trait_def_id, &paths::SERDE_DESERIALIZE);
        if let ty::Adt(def, _) = ty.kind();
        if let Some(local_def_id) = def.did().as_local();
        let adt_hir_id = cx.tcx.hir().local_def_id_to_hir_id(local_def_id);
        if !is_lint_allowed(cx, UNSAFE_DERIVE_DESERIALIZE, adt_hir_id);
        if cx.tcx.inherent_impls(def.did())
            .iter()
            .map(|imp_did| cx.tcx.hir().expect_item(imp_did.expect_local()))
            .any(|imp| has_unsafe(cx, imp));
        then {
            span_lint_and_help(
                cx,
                UNSAFE_DERIVE_DESERIALIZE,
                item.span,
                "you are deriving `serde::Deserialize` on a type that has methods using `unsafe`",
                None,
                "consider implementing `serde::Deserialize` manually. See https://serde.rs/impl-deserialize.html"
            );
        }
    }
}

struct UnsafeVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    has_unsafe: bool,
}

impl<'tcx> Visitor<'tcx> for UnsafeVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn visit_fn(&mut self, kind: FnKind<'tcx>, decl: &'tcx FnDecl<'_>, body_id: BodyId, _: Span, id: LocalDefId) {
        if self.has_unsafe {
            return;
        }

        if_chain! {
            if let Some(header) = kind.header();
            if header.unsafety == Unsafety::Unsafe;
            then {
                self.has_unsafe = true;
            }
        }

        walk_fn(self, kind, decl, body_id, id);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.has_unsafe {
            return;
        }

        if let ExprKind::Block(block, _) = expr.kind {
            if block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) {
                self.has_unsafe = true;
            }
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}

/// Implementation of the `DERIVE_PARTIAL_EQ_WITHOUT_EQ` lint.
fn check_partial_eq_without_eq<'tcx>(cx: &LateContext<'tcx>, span: Span, trait_ref: &hir::TraitRef<'_>, ty: Ty<'tcx>) {
    if_chain! {
        if let ty::Adt(adt, substs) = ty.kind();
        if cx.tcx.visibility(adt.did()).is_public();
        if let Some(eq_trait_def_id) = cx.tcx.get_diagnostic_item(sym::Eq);
        if let Some(def_id) = trait_ref.trait_def_id();
        if cx.tcx.is_diagnostic_item(sym::PartialEq, def_id);
        let param_env = param_env_for_derived_eq(cx.tcx, adt.did(), eq_trait_def_id);
        if !implements_trait_with_env(cx.tcx, param_env, ty, eq_trait_def_id, []);
        // If all of our fields implement `Eq`, we can implement `Eq` too
        if adt
            .all_fields()
            .map(|f| f.ty(cx.tcx, substs))
            .all(|ty| implements_trait_with_env(cx.tcx, param_env, ty, eq_trait_def_id, []));
        then {
            span_lint_and_sugg(
                cx,
                DERIVE_PARTIAL_EQ_WITHOUT_EQ,
                span.ctxt().outer_expn_data().call_site,
                "you are deriving `PartialEq` and can implement `Eq`",
                "consider deriving `Eq` as well",
                "PartialEq, Eq".to_string(),
                Applicability::MachineApplicable,
            )
        }
    }
}

/// Creates the `ParamEnv` used for the give type's derived `Eq` impl.
fn param_env_for_derived_eq(tcx: TyCtxt<'_>, did: DefId, eq_trait_id: DefId) -> ParamEnv<'_> {
    // Initial map from generic index to param def.
    // Vec<(param_def, needs_eq)>
    let mut params = tcx
        .generics_of(did)
        .params
        .iter()
        .map(|p| (p, matches!(p.kind, GenericParamDefKind::Type { .. })))
        .collect::<Vec<_>>();

    let ty_predicates = tcx.predicates_of(did).predicates;
    for (p, _) in ty_predicates {
        if let PredicateKind::Clause(Clause::Trait(p)) = p.kind().skip_binder()
            && p.trait_ref.def_id == eq_trait_id
            && let ty::Param(self_ty) = p.trait_ref.self_ty().kind()
            && p.constness == BoundConstness::NotConst
        {
            // Flag types which already have an `Eq` bound.
            params[self_ty.index as usize].1 = false;
        }
    }

    ParamEnv::new(
        tcx.mk_predicates(ty_predicates.iter().map(|&(p, _)| p).chain(
            params.iter().filter(|&&(_, needs_eq)| needs_eq).map(|&(param, _)| {
                tcx.mk_predicate(Binder::dummy(PredicateKind::Clause(Clause::Trait(TraitPredicate {
                    trait_ref: tcx.mk_trait_ref(eq_trait_id, [tcx.mk_param_from_def(param)]),
                    constness: BoundConstness::NotConst,
                    polarity: ImplPolarity::Positive,
                }))))
            }),
        )),
        Reveal::UserFacing,
        Constness::NotConst,
    )
}
