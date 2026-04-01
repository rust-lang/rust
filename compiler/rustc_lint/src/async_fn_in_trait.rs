use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_trait_selection::error_reporting::traits::suggestions::suggest_desugaring_async_fn_to_impl_future_in_trait;

use crate::lints::AsyncFnInTraitDiag;
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `async_fn_in_trait` lint detects use of `async fn` in the
    /// definition of a publicly-reachable trait.
    ///
    /// ### Example
    ///
    /// ```rust
    /// pub trait Trait {
    ///     async fn method(&self);
    /// }
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When `async fn` is used in a trait definition, the trait does not
    /// promise that the opaque [`Future`] returned by the associated function
    /// or method will implement any [auto traits] such as [`Send`]. This may
    /// be surprising and may make the associated functions or methods on the
    /// trait less useful than intended. On traits exposed publicly from a
    /// crate, this may affect downstream crates whose authors cannot alter
    /// the trait definition.
    ///
    /// For example, this code is invalid:
    ///
    /// ```rust,compile_fail
    /// pub trait Trait {
    ///     async fn method(&self) {}
    /// }
    ///
    /// fn test<T: Trait>(x: T) {
    ///     fn spawn<T: Send>(_: T) {}
    ///     spawn(x.method()); // Not OK.
    /// }
    /// ```
    ///
    /// This lint exists to warn authors of publicly-reachable traits that
    /// they may want to consider desugaring the `async fn` to a normal `fn`
    /// that returns an opaque `impl Future<..> + Send` type.
    ///
    /// For example, instead of:
    ///
    /// ```rust
    /// pub trait Trait {
    ///     async fn method(&self) {}
    /// }
    /// ```
    ///
    /// The author of the trait may want to write:
    ///
    ///
    /// ```rust
    /// use core::future::Future;
    /// pub trait Trait {
    ///     fn method(&self) -> impl Future<Output = ()> + Send { async {} }
    /// }
    /// ```
    ///
    /// This still allows the use of `async fn` within impls of the trait.
    /// However, it also means that the trait will never be compatible with
    /// impls where the returned [`Future`] of the method does not implement
    /// `Send`.
    ///
    /// Conversely, if the trait is used only locally, if it is never used in
    /// generic functions, or if it is only used in single-threaded contexts
    /// that do not care whether the returned [`Future`] implements [`Send`],
    /// then the lint may be suppressed.
    ///
    /// [`Future`]: https://doc.rust-lang.org/core/future/trait.Future.html
    /// [`Send`]: https://doc.rust-lang.org/core/marker/trait.Send.html
    /// [auto traits]: https://doc.rust-lang.org/reference/special-types-and-traits.html#auto-traits
    pub ASYNC_FN_IN_TRAIT,
    Warn,
    "use of `async fn` in definition of a publicly-reachable trait"
}

declare_lint_pass!(
    /// Lint for use of `async fn` in the definition of a publicly-reachable
    /// trait.
    AsyncFnInTrait => [ASYNC_FN_IN_TRAIT]
);

impl<'tcx> LateLintPass<'tcx> for AsyncFnInTrait {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'tcx>) {
        if let hir::TraitItemKind::Fn(sig, body) = item.kind
            && let hir::IsAsync::Async(async_span) = sig.header.asyncness
        {
            // RTN can be used to bound `async fn` in traits in a better way than "always"
            if cx.tcx.features().return_type_notation() {
                return;
            }

            // Only need to think about library implications of reachable traits
            if !cx.tcx.effective_visibilities(()).is_reachable(item.owner_id.def_id) {
                return;
            }

            let hir::FnRetTy::Return(hir::Ty {
                kind: hir::TyKind::OpaqueDef(opaq_def, ..), ..
            }) = sig.decl.output
            else {
                // This should never happen, but let's not ICE.
                return;
            };
            let sugg = suggest_desugaring_async_fn_to_impl_future_in_trait(
                cx.tcx,
                sig,
                body,
                opaq_def.def_id,
                " + Send",
            );
            cx.tcx.emit_node_span_lint(
                ASYNC_FN_IN_TRAIT,
                item.hir_id(),
                async_span,
                AsyncFnInTraitDiag { sugg },
            );
        }
    }
}
