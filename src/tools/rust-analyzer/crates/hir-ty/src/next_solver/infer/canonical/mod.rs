//! **Canonicalization** is the key to constructing a query in the
//! middle of type inference. Ordinarily, it is not possible to store
//! types from type inference in query keys, because they contain
//! references to inference variables whose lifetimes are too short
//! and so forth. Canonicalizing a value T1 using `canonicalize_query`
//! produces two things:
//!
//! - a value T2 where each unbound inference variable has been
//!   replaced with a **canonical variable**;
//! - a map M (of type `CanonicalVarValues`) from those canonical
//!   variables back to the original.
//!
//! We can then do queries using T2. These will give back constraints
//! on the canonical variables which can be translated, using the map
//! M, into constraints in our source context. This process of
//! translating the results back is done by the
//! `instantiate_query_result` method.
//!
//! For a more detailed look at what is happening here, check
//! out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use crate::next_solver::{
    AliasTy, Binder, Canonical, CanonicalVarValues, CanonicalVars, Const, DbInterner, GenericArg,
    Goal, ParamEnv, PlaceholderConst, PlaceholderRegion, PlaceholderTy, Predicate, PredicateKind,
    Region, Ty, TyKind,
    infer::{
        DefineOpaqueTypes, InferCtxt, TypeTrace,
        traits::{Obligation, PredicateObligations},
    },
};
use instantiate::CanonicalExt;
use rustc_index::IndexVec;
use rustc_type_ir::{
    AliasRelationDirection, AliasTyKind, CanonicalTyVarKind, CanonicalVarKind, InferTy,
    TypeFoldable, UniverseIndex, Upcast, Variance,
    inherent::{SliceLike, Ty as _},
    relate::{
        Relate, TypeRelation, VarianceDiagInfo,
        combine::{super_combine_consts, super_combine_tys},
    },
};

pub mod instantiate;

impl<'db> InferCtxt<'db> {
    /// Creates an instantiation S for the canonical value with fresh inference
    /// variables and placeholders then applies it to the canonical value.
    /// Returns both the instantiated result *and* the instantiation S.
    ///
    /// This can be invoked as part of constructing an
    /// inference context at the start of a query (see
    /// `InferCtxtBuilder::build_with_canonical`). It basically
    /// brings the canonical value "into scope" within your new infcx.
    ///
    /// At the end of processing, the instantiation S (once
    /// canonicalized) then represents the values that you computed
    /// for each of the canonical inputs to your query.
    pub fn instantiate_canonical<T>(
        &self,
        canonical: &Canonical<'db, T>,
    ) -> (T, CanonicalVarValues<'db>)
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        // For each universe that is referred to in the incoming
        // query, create a universe in our local inference context. In
        // practice, as of this writing, all queries have no universes
        // in them, so this code has no effect, but it is looking
        // forward to the day when we *do* want to carry universes
        // through into queries.
        //
        // Instantiate the root-universe content into the current universe,
        // and create fresh universes for the higher universes.
        let universes: IndexVec<UniverseIndex, _> = std::iter::once(self.universe())
            .chain((1..=canonical.max_universe.as_u32()).map(|_| self.create_next_universe()))
            .collect();

        let canonical_inference_vars =
            self.instantiate_canonical_vars(canonical.variables, |ui| universes[ui]);
        let result = canonical.instantiate(self.interner, &canonical_inference_vars);
        (result, canonical_inference_vars)
    }

    /// Given the "infos" about the canonical variables from some
    /// canonical, creates fresh variables with the same
    /// characteristics (see `instantiate_canonical_var` for
    /// details). You can then use `instantiate` to instantiate the
    /// canonical variable with these inference variables.
    fn instantiate_canonical_vars(
        &self,
        variables: CanonicalVars<'db>,
        universe_map: impl Fn(UniverseIndex) -> UniverseIndex,
    ) -> CanonicalVarValues<'db> {
        CanonicalVarValues {
            var_values: self.interner.mk_args_from_iter(
                variables.iter().map(|info| self.instantiate_canonical_var(info, &universe_map)),
            ),
        }
    }

    /// Given the "info" about a canonical variable, creates a fresh
    /// variable for it. If this is an existentially quantified
    /// variable, then you'll get a new inference variable; if it is a
    /// universally quantified variable, you get a placeholder.
    ///
    /// FIXME(-Znext-solver): This is public because it's used by the
    /// new trait solver which has a different canonicalization routine.
    /// We should somehow deduplicate all of this.
    pub fn instantiate_canonical_var(
        &self,
        cv_info: CanonicalVarKind<DbInterner<'db>>,
        universe_map: impl Fn(UniverseIndex) -> UniverseIndex,
    ) -> GenericArg<'db> {
        match cv_info {
            CanonicalVarKind::Ty(ty_kind) => {
                let ty = match ty_kind {
                    CanonicalTyVarKind::General(ui) => {
                        self.next_ty_var_in_universe(universe_map(ui))
                    }

                    CanonicalTyVarKind::Int => self.next_int_var(),

                    CanonicalTyVarKind::Float => self.next_float_var(),
                };
                ty.into()
            }

            CanonicalVarKind::PlaceholderTy(PlaceholderTy { universe, bound }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = PlaceholderTy { universe: universe_mapped, bound };
                Ty::new_placeholder(self.interner, placeholder_mapped).into()
            }

            CanonicalVarKind::Region(ui) => {
                self.next_region_var_in_universe(universe_map(ui)).into()
            }

            CanonicalVarKind::PlaceholderRegion(PlaceholderRegion { universe, bound }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped: crate::next_solver::Placeholder<
                    crate::next_solver::BoundRegion,
                > = PlaceholderRegion { universe: universe_mapped, bound };
                Region::new_placeholder(self.interner, placeholder_mapped).into()
            }

            CanonicalVarKind::Const(ui) => self.next_const_var_in_universe(universe_map(ui)).into(),
            CanonicalVarKind::PlaceholderConst(PlaceholderConst { universe, bound }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = PlaceholderConst { universe: universe_mapped, bound };
                Const::new_placeholder(self.interner, placeholder_mapped).into()
            }
        }
    }
}
