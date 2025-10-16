use clippy_utils::res::MaybeResPath;
use rustc_hir::def::Res;
use rustc_hir::{Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

mod derive_ord_xor_partial_ord;
mod derive_partial_eq_without_eq;
mod derived_hash_with_manual_eq;
mod expl_impl_clone_on_copy;
mod unsafe_derive_deserialize;

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
            self_ty,
            ..
        }) = item.kind
            && let Res::Def(_, def_id) = *self_ty.basic_res()
            && let Some(local_def_id) = def_id.as_local()
        {
            let adt_hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id);
            let trait_ref = &of_trait.trait_ref;
            let ty = cx.tcx.type_of(item.owner_id).instantiate_identity();
            let is_automatically_derived = cx.tcx.is_automatically_derived(item.owner_id.to_def_id());

            derived_hash_with_manual_eq::check(cx, item.span, trait_ref, ty, adt_hir_id, is_automatically_derived);
            derive_ord_xor_partial_ord::check(cx, item.span, trait_ref, ty, adt_hir_id, is_automatically_derived);

            if is_automatically_derived {
                unsafe_derive_deserialize::check(cx, item, trait_ref, ty, adt_hir_id);
                derive_partial_eq_without_eq::check(cx, item.span, trait_ref, ty, adt_hir_id);
            } else {
                expl_impl_clone_on_copy::check(cx, item, trait_ref, ty, adt_hir_id);
            }
        }
    }
}
