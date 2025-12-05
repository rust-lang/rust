//! Helper routines for higher-ranked things. See the `doc` module at
//! the end of the file for details.

use rustc_type_ir::TypeFoldable;
use tracing::{debug, instrument};

use crate::next_solver::fold::FnMutDelegate;
use crate::next_solver::infer::InferCtxt;
use crate::next_solver::{
    Binder, BoundConst, BoundRegion, BoundTy, Const, DbInterner, PlaceholderConst,
    PlaceholderRegion, PlaceholderTy, Region, Ty,
};

impl<'db> InferCtxt<'db> {
    /// Replaces all bound variables (lifetimes, types, and constants) bound by
    /// `binder` with placeholder variables in a new universe. This means that the
    /// new placeholders can only be named by inference variables created after
    /// this method has been called.
    ///
    /// This is the first step of checking subtyping when higher-ranked things are involved.
    /// For more details visit the relevant sections of the [rustc dev guide].
    ///
    /// `fn enter_forall` should be preferred over this method.
    ///
    /// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
    #[instrument(level = "debug", skip(self), ret)]
    pub fn enter_forall_and_leak_universe<T>(&self, binder: Binder<'db, T>) -> T
    where
        T: TypeFoldable<DbInterner<'db>> + Clone,
    {
        if let Some(inner) = binder.clone().no_bound_vars() {
            return inner;
        }

        let next_universe = self.create_next_universe();

        let delegate = FnMutDelegate {
            regions: &mut |br: BoundRegion| {
                Region::new_placeholder(
                    self.interner,
                    PlaceholderRegion { universe: next_universe, bound: br },
                )
            },
            types: &mut |bound_ty: BoundTy| {
                Ty::new_placeholder(
                    self.interner,
                    PlaceholderTy { universe: next_universe, bound: bound_ty },
                )
            },
            consts: &mut |bound: BoundConst| {
                Const::new_placeholder(
                    self.interner,
                    PlaceholderConst { universe: next_universe, bound },
                )
            },
        };

        debug!(?next_universe);
        self.interner.replace_bound_vars_uncached(binder, delegate)
    }

    /// Replaces all bound variables (lifetimes, types, and constants) bound by
    /// `binder` with placeholder variables in a new universe and then calls the
    /// closure `f` with the instantiated value. The new placeholders can only be
    /// named by inference variables created inside of the closure `f` or afterwards.
    ///
    /// This is the first step of checking subtyping when higher-ranked things are involved.
    /// For more details visit the relevant sections of the [rustc dev guide].
    ///
    /// This method should be preferred over `fn enter_forall_and_leak_universe`.
    ///
    /// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
    #[instrument(level = "debug", skip(self, f))]
    pub fn enter_forall<T, U>(&self, forall: Binder<'db, T>, f: impl FnOnce(T) -> U) -> U
    where
        T: TypeFoldable<DbInterner<'db>> + Clone,
    {
        // FIXME: currently we do nothing to prevent placeholders with the new universe being
        // used after exiting `f`. For example region subtyping can result in outlives constraints
        // that name placeholders created in this function. Nested goals from type relations can
        // also contain placeholders created by this function.
        let value = self.enter_forall_and_leak_universe(forall);
        debug!(?value);
        f(value)
    }
}
