use crate::utils::paths;
use crate::utils::{
    get_trait_def_id, is_allowed, is_automatically_derived, is_copy, match_def_path, match_path, span_lint_and_help,
    span_lint_and_note, span_lint_and_then,
};
use if_chain::if_chain;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{walk_expr, walk_fn, walk_item, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::{
    BlockCheckMode, BodyId, Expr, ExprKind, FnDecl, HirId, Impl, Item, ItemKind, TraitRef, UnsafeSource, Unsafety,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for deriving `Hash` but implementing `PartialEq`
    /// explicitly or vice versa.
    ///
    /// **Why is this bad?** The implementation of these traits must agree (for
    /// example for use with `HashMap`) so it’s probably a bad idea to use a
    /// default-generated `Hash` implementation with an explicitly defined
    /// `PartialEq`. In particular, the following must hold for any type:
    ///
    /// ```text
    /// k1 == k2 ⇒ hash(k1) == hash(k2)
    /// ```
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// #[derive(Hash)]
    /// struct Foo;
    ///
    /// impl PartialEq for Foo {
    ///     ...
    /// }
    /// ```
    pub DERIVE_HASH_XOR_EQ,
    correctness,
    "deriving `Hash` but implementing `PartialEq` explicitly"
}

declare_clippy_lint! {
    /// **What it does:** Checks for deriving `Ord` but implementing `PartialOrd`
    /// explicitly or vice versa.
    ///
    /// **Why is this bad?** The implementation of these traits must agree (for
    /// example for use with `sort`) so it’s probably a bad idea to use a
    /// default-generated `Ord` implementation with an explicitly defined
    /// `PartialOrd`. In particular, the following must hold for any type
    /// implementing `Ord`:
    ///
    /// ```text
    /// k1.cmp(&k2) == k1.partial_cmp(&k2).unwrap()
    /// ```
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
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
    pub DERIVE_ORD_XOR_PARTIAL_ORD,
    correctness,
    "deriving `Ord` but implementing `PartialOrd` explicitly"
}

declare_clippy_lint! {
    /// **What it does:** Checks for explicit `Clone` implementations for `Copy`
    /// types.
    ///
    /// **Why is this bad?** To avoid surprising behaviour, these traits should
    /// agree and the behaviour of `Copy` cannot be overridden. In almost all
    /// situations a `Copy` type should have a `Clone` implementation that does
    /// nothing more than copy the object, which is what `#[derive(Copy, Clone)]`
    /// gets you.
    ///
    /// **Known problems:** Bounds of generic types are sometimes wrong: https://github.com/rust-lang/rust/issues/26925
    ///
    /// **Example:**
    /// ```rust,ignore
    /// #[derive(Copy)]
    /// struct Foo;
    ///
    /// impl Clone for Foo {
    ///     // ..
    /// }
    /// ```
    pub EXPL_IMPL_CLONE_ON_COPY,
    pedantic,
    "implementing `Clone` explicitly on `Copy` types"
}

declare_clippy_lint! {
    /// **What it does:** Checks for deriving `serde::Deserialize` on a type that
    /// has methods using `unsafe`.
    ///
    /// **Why is this bad?** Deriving `serde::Deserialize` will create a constructor
    /// that may violate invariants hold by another constructor.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
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
    pub UNSAFE_DERIVE_DESERIALIZE,
    pedantic,
    "deriving `serde::Deserialize` on a type that has methods using `unsafe`"
}

declare_lint_pass!(Derive => [
    EXPL_IMPL_CLONE_ON_COPY,
    DERIVE_HASH_XOR_EQ,
    DERIVE_ORD_XOR_PARTIAL_ORD,
    UNSAFE_DERIVE_DESERIALIZE
]);

impl<'tcx> LateLintPass<'tcx> for Derive {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Impl(Impl {
            of_trait: Some(ref trait_ref),
            ..
        }) = item.kind
        {
            let ty = cx.tcx.type_of(cx.tcx.hir().local_def_id(item.hir_id));
            let is_automatically_derived = is_automatically_derived(&*item.attrs);

            check_hash_peq(cx, item.span, trait_ref, ty, is_automatically_derived);
            check_ord_partial_ord(cx, item.span, trait_ref, ty, is_automatically_derived);

            if is_automatically_derived {
                check_unsafe_derive_deserialize(cx, item, trait_ref, ty);
            } else {
                check_copy_clone(cx, item, trait_ref, ty);
            }
        }
    }
}

/// Implementation of the `DERIVE_HASH_XOR_EQ` lint.
fn check_hash_peq<'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    trait_ref: &TraitRef<'_>,
    ty: Ty<'tcx>,
    hash_is_automatically_derived: bool,
) {
    if_chain! {
        if let Some(peq_trait_def_id) = cx.tcx.lang_items().eq_trait();
        if let Some(def_id) = trait_ref.trait_def_id();
        if match_def_path(cx, def_id, &paths::HASH);
        then {
            // Look for the PartialEq implementations for `ty`
            cx.tcx.for_each_relevant_impl(peq_trait_def_id, ty, |impl_id| {
                let peq_is_automatically_derived = is_automatically_derived(&cx.tcx.get_attrs(impl_id));

                if peq_is_automatically_derived == hash_is_automatically_derived {
                    return;
                }

                let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

                // Only care about `impl PartialEq<Foo> for Foo`
                // For `impl PartialEq<B> for A, input_types is [A, B]
                if trait_ref.substs.type_at(1) == ty {
                    let mess = if peq_is_automatically_derived {
                        "you are implementing `Hash` explicitly but have derived `PartialEq`"
                    } else {
                        "you are deriving `Hash` but have implemented `PartialEq` explicitly"
                    };

                    span_lint_and_then(
                        cx,
                        DERIVE_HASH_XOR_EQ,
                        span,
                        mess,
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
    trait_ref: &TraitRef<'_>,
    ty: Ty<'tcx>,
    ord_is_automatically_derived: bool,
) {
    if_chain! {
        if let Some(ord_trait_def_id) = get_trait_def_id(cx, &paths::ORD);
        if let Some(partial_ord_trait_def_id) = cx.tcx.lang_items().partial_ord_trait();
        if let Some(def_id) = &trait_ref.trait_def_id();
        if *def_id == ord_trait_def_id;
        then {
            // Look for the PartialOrd implementations for `ty`
            cx.tcx.for_each_relevant_impl(partial_ord_trait_def_id, ty, |impl_id| {
                let partial_ord_is_automatically_derived = is_automatically_derived(&cx.tcx.get_attrs(impl_id));

                if partial_ord_is_automatically_derived == ord_is_automatically_derived {
                    return;
                }

                let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

                // Only care about `impl PartialOrd<Foo> for Foo`
                // For `impl PartialOrd<B> for A, input_types is [A, B]
                if trait_ref.substs.type_at(1) == ty {
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
fn check_copy_clone<'tcx>(cx: &LateContext<'tcx>, item: &Item<'_>, trait_ref: &TraitRef<'_>, ty: Ty<'tcx>) {
    if match_path(&trait_ref.path, &paths::CLONE_TRAIT) {
        if !is_copy(cx, ty) {
            return;
        }

        match *ty.kind() {
            ty::Adt(def, _) if def.is_union() => return,

            // Some types are not Clone by default but could be cloned “by hand” if necessary
            ty::Adt(def, substs) => {
                for variant in &def.variants {
                    for field in &variant.fields {
                        if let ty::FnDef(..) = field.ty(cx.tcx, substs).kind() {
                            return;
                        }
                    }
                    for subst in substs {
                        if let ty::subst::GenericArgKind::Type(subst) = subst.unpack() {
                            if let ty::Param(_) = subst.kind() {
                                return;
                            }
                        }
                    }
                }
            },
            _ => (),
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
}

/// Implementation of the `UNSAFE_DERIVE_DESERIALIZE` lint.
fn check_unsafe_derive_deserialize<'tcx>(
    cx: &LateContext<'tcx>,
    item: &Item<'_>,
    trait_ref: &TraitRef<'_>,
    ty: Ty<'tcx>,
) {
    fn item_from_def_id<'tcx>(cx: &LateContext<'tcx>, def_id: DefId) -> &'tcx Item<'tcx> {
        let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        cx.tcx.hir().expect_item(hir_id)
    }

    fn has_unsafe<'tcx>(cx: &LateContext<'tcx>, item: &'tcx Item<'_>) -> bool {
        let mut visitor = UnsafeVisitor { cx, has_unsafe: false };
        walk_item(&mut visitor, item);
        visitor.has_unsafe
    }

    if_chain! {
        if let Some(trait_def_id) = trait_ref.trait_def_id();
        if match_def_path(cx, trait_def_id, &paths::SERDE_DESERIALIZE);
        if let ty::Adt(def, _) = ty.kind();
        if let Some(local_def_id) = def.did.as_local();
        let adt_hir_id = cx.tcx.hir().local_def_id_to_hir_id(local_def_id);
        if !is_allowed(cx, UNSAFE_DERIVE_DESERIALIZE, adt_hir_id);
        if cx.tcx.inherent_impls(def.did)
            .iter()
            .map(|imp_did| item_from_def_id(cx, *imp_did))
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
    type Map = Map<'tcx>;

    fn visit_fn(&mut self, kind: FnKind<'tcx>, decl: &'tcx FnDecl<'_>, body_id: BodyId, span: Span, id: HirId) {
        if self.has_unsafe {
            return;
        }

        if_chain! {
            if let Some(header) = kind.header();
            if let Unsafety::Unsafe = header.unsafety;
            then {
                self.has_unsafe = true;
            }
        }

        walk_fn(self, kind, decl, body_id, span, id);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.has_unsafe {
            return;
        }

        if let ExprKind::Block(block, _) = expr.kind {
            match block.rules {
                BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
                | BlockCheckMode::PushUnsafeBlock(UnsafeSource::UserProvided)
                | BlockCheckMode::PopUnsafeBlock(UnsafeSource::UserProvided) => {
                    self.has_unsafe = true;
                },
                _ => {},
            }
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.cx.tcx.hir())
    }
}
