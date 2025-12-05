use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::implements_trait;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::declare_lint_pass;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// This lint is concerned with the semantics of `Borrow` and `Hash` for a
    /// type that implements all three of `Hash`, `Borrow<str>` and `Borrow<[u8]>`
    /// as it is impossible to satisfy the semantics of Borrow and `Hash` for
    /// both `Borrow<str>` and `Borrow<[u8]>`.
    ///
    /// ### Why is this bad?
    ///
    /// When providing implementations for `Borrow<T>`, one should consider whether the different
    /// implementations should act as facets or representations of the underlying type. Generic code
    /// typically uses `Borrow<T>` when it relies on the identical behavior of these additional trait
    /// implementations. These traits will likely appear as additional trait bounds.
    ///
    /// In particular `Eq`, `Ord` and `Hash` must be equivalent for borrowed and owned values:
    /// `x.borrow() == y.borrow()` should give the same result as `x == y`.
    /// It follows then that the following equivalence must hold:
    /// `hash(x) == hash((x as Borrow<[u8]>).borrow()) == hash((x as Borrow<str>).borrow())`
    ///
    /// Unfortunately it doesn't hold as `hash("abc") != hash("abc".as_bytes())`.
    /// This happens because the `Hash` impl for str passes an additional `0xFF` byte to
    /// the hasher to avoid collisions. For example, given the tuples `("a", "bc")`, and `("ab", "c")`,
    /// the two tuples would have the same hash value if the `0xFF` byte was not added.
    ///
    /// ### Example
    ///
    /// ```
    /// use std::borrow::Borrow;
    /// use std::hash::{Hash, Hasher};
    ///
    /// struct ExampleType {
    ///     data: String
    /// }
    ///
    /// impl Hash for ExampleType {
    ///     fn hash<H: Hasher>(&self, state: &mut H) {
    ///         self.data.hash(state);
    ///     }
    /// }
    ///
    /// impl Borrow<str> for ExampleType {
    ///     fn borrow(&self) -> &str {
    ///         &self.data
    ///     }
    /// }
    ///
    /// impl Borrow<[u8]> for ExampleType {
    ///     fn borrow(&self) -> &[u8] {
    ///         self.data.as_bytes()
    ///     }
    /// }
    /// ```
    /// As a consequence, hashing a `&ExampleType` and hashing the result of the two
    /// borrows will result in different values.
    ///
    #[clippy::version = "1.76.0"]
    pub IMPL_HASH_BORROW_WITH_STR_AND_BYTES,
    correctness,
    "ensures that the semantics of `Borrow` for `Hash` are satisfied when `Borrow<str>` and `Borrow<[u8]>` are implemented"
}

declare_lint_pass!(ImplHashWithBorrowStrBytes => [IMPL_HASH_BORROW_WITH_STR_AND_BYTES]);

impl LateLintPass<'_> for ImplHashWithBorrowStrBytes {
    /// We are emitting this lint at the Hash impl of a type that implements all
    /// three of `Hash`, `Borrow<str>` and `Borrow<[u8]>`.
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if let ItemKind::Impl(imp) = item.kind
            && let Some(of_trait) = imp.of_trait
            && let ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
            && let Some(hash_id) = cx.tcx.get_diagnostic_item(sym::Hash)
            && Res::Def(DefKind::Trait, hash_id) == of_trait.trait_ref.path.res
            && let Some(borrow_id) = cx.tcx.get_diagnostic_item(sym::Borrow)
            // since we are in the `Hash` impl, we don't need to check for that.
            // we need only to check for `Borrow<str>` and `Borrow<[u8]>`
            && implements_trait(cx, ty, borrow_id, &[cx.tcx.types.str_.into()])
            && implements_trait(cx, ty, borrow_id, &[Ty::new_slice(cx.tcx, cx.tcx.types.u8).into()])
        {
            span_lint_and_then(
                cx,
                IMPL_HASH_BORROW_WITH_STR_AND_BYTES,
                of_trait.trait_ref.path.span,
                "the semantics of `Borrow<T>` around `Hash` can't be satisfied when both `Borrow<str>` and `Borrow<[u8]>` are implemented",
                |diag| {
                    diag.note("the `Borrow` semantics require that `Hash` must behave the same for all implementations of Borrow<T>");
                    diag.note(
          "however, the hash implementations of a string (`str`) and the bytes of a string `[u8]` do not behave the same ..."
      );
                    diag.note("... as (`hash(\"abc\") != hash(\"abc\".as_bytes())`");
                    diag.help("consider either removing one of the  `Borrow` implementations (`Borrow<str>` or `Borrow<[u8]>`) ...");
                    diag.help("... or not implementing `Hash` for this type");
                },
            );
        }
    }
}
