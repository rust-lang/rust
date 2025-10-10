//! This module contains code to canonicalize values into a `Canonical<'db, T>`.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use rustc_hash::FxHashMap;
use rustc_index::Idx;
use rustc_type_ir::InferTy::{self, FloatVar, IntVar, TyVar};
use rustc_type_ir::inherent::{Const as _, IntoKind as _, Region as _, SliceLike, Ty as _};
use rustc_type_ir::{
    BoundVar, CanonicalQueryInput, DebruijnIndex, Flags, InferConst, RegionKind, TyVid, TypeFlags,
    TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt, UniverseIndex,
};
use smallvec::SmallVec;
use tracing::debug;

use crate::next_solver::infer::InferCtxt;
use crate::next_solver::{
    Binder, BoundConst, BoundRegion, BoundRegionKind, BoundTy, Canonical, CanonicalVarKind,
    CanonicalVars, Const, ConstKind, DbInterner, GenericArg, ParamEnvAnd, Placeholder, Region, Ty,
    TyKind,
};

/// When we canonicalize a value to form a query, we wind up replacing
/// various parts of it with canonical variables. This struct stores
/// those replaced bits to remember for when we process the query
/// result.
#[derive(Clone, Debug)]
pub struct OriginalQueryValues<'db> {
    /// Map from the universes that appear in the query to the universes in the
    /// caller context. For all queries except `evaluate_goal` (used by Chalk),
    /// we only ever put ROOT values into the query, so this map is very
    /// simple.
    pub universe_map: SmallVec<[UniverseIndex; 4]>,

    /// This is equivalent to `CanonicalVarValues`, but using a
    /// `SmallVec` yields a significant performance win.
    pub var_values: SmallVec<[GenericArg<'db>; 8]>,
}

impl<'db> Default for OriginalQueryValues<'db> {
    fn default() -> Self {
        let mut universe_map = SmallVec::default();
        universe_map.push(UniverseIndex::ROOT);

        Self { universe_map, var_values: SmallVec::default() }
    }
}

impl<'db> InferCtxt<'db> {
    /// Canonicalizes a query value `V`. When we canonicalize a query,
    /// we not only canonicalize unbound inference variables, but we
    /// *also* replace all free regions whatsoever. So for example a
    /// query like `T: Trait<'static>` would be canonicalized to
    ///
    /// ```text
    /// T: Trait<'?0>
    /// ```
    ///
    /// with a mapping M that maps `'?0` to `'static`.
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc dev guide][c].
    ///
    /// [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html#canonicalizing-the-query
    pub fn canonicalize_query<V>(
        &self,
        value: ParamEnvAnd<'db, V>,
        query_state: &mut OriginalQueryValues<'db>,
    ) -> CanonicalQueryInput<DbInterner<'db>, ParamEnvAnd<'db, V>>
    where
        V: TypeFoldable<DbInterner<'db>>,
    {
        let (param_env, value) = value.into_parts();
        // FIXME(#118965): We don't canonicalize the static lifetimes that appear in the
        // `param_env` because they are treated differently by trait selection.
        let canonical_param_env = Canonicalizer::canonicalize(
            param_env,
            self,
            self.interner,
            &CanonicalizeFreeRegionsOtherThanStatic,
            query_state,
        );

        let canonical = Canonicalizer::canonicalize_with_base(
            canonical_param_env,
            value,
            self,
            self.interner,
            &CanonicalizeAllFreeRegions,
            query_state,
        )
        .unchecked_map(|(param_env, value)| ParamEnvAnd { param_env, value });
        CanonicalQueryInput { canonical, typing_mode: self.typing_mode() }
    }

    /// Canonicalizes a query *response* `V`. When we canonicalize a
    /// query response, we only canonicalize unbound inference
    /// variables, and we leave other free regions alone. So,
    /// continuing with the example from `canonicalize_query`, if
    /// there was an input query `T: Trait<'static>`, it would have
    /// been canonicalized to
    ///
    /// ```text
    /// T: Trait<'?0>
    /// ```
    ///
    /// with a mapping M that maps `'?0` to `'static`. But if we found that there
    /// exists only one possible impl of `Trait`, and it looks like
    /// ```ignore (illustrative)
    /// impl<T> Trait<'static> for T { .. }
    /// ```
    /// then we would prepare a query result R that (among other
    /// things) includes a mapping to `'?0 := 'static`. When
    /// canonicalizing this query result R, we would leave this
    /// reference to `'static` alone.
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc dev guide][c].
    ///
    /// [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html#canonicalizing-the-query-result
    pub fn canonicalize_response<V>(&self, value: V) -> Canonical<'db, V>
    where
        V: TypeFoldable<DbInterner<'db>>,
    {
        let mut query_state = OriginalQueryValues::default();
        Canonicalizer::canonicalize(
            value,
            self,
            self.interner,
            &CanonicalizeQueryResponse,
            &mut query_state,
        )
    }

    pub fn canonicalize_user_type_annotation<V>(&self, value: V) -> Canonical<'db, V>
    where
        V: TypeFoldable<DbInterner<'db>>,
    {
        let mut query_state = OriginalQueryValues::default();
        Canonicalizer::canonicalize(
            value,
            self,
            self.interner,
            &CanonicalizeUserTypeAnnotation,
            &mut query_state,
        )
    }
}

/// Controls how we canonicalize "free regions" that are not inference
/// variables. This depends on what we are canonicalizing *for* --
/// e.g., if we are canonicalizing to create a query, we want to
/// replace those with inference variables, since we want to make a
/// maximally general query. But if we are canonicalizing a *query
/// response*, then we don't typically replace free regions, as they
/// must have been introduced from other parts of the system.
trait CanonicalizeMode {
    fn canonicalize_free_region<'db>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'db>,
        r: Region<'db>,
    ) -> Region<'db>;

    fn any(&self) -> bool;

    // Do we preserve universe of variables.
    fn preserve_universes(&self) -> bool;
}

struct CanonicalizeQueryResponse;

impl CanonicalizeMode for CanonicalizeQueryResponse {
    fn canonicalize_free_region<'db>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'db>,
        mut r: Region<'db>,
    ) -> Region<'db> {
        let infcx = canonicalizer.infcx;

        if let RegionKind::ReVar(vid) = r.kind() {
            r = infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(canonicalizer.tcx, vid);
            debug!(
                "canonical: region var found with vid {vid:?}, \
                     opportunistically resolved to {r:?}",
            );
        };

        match r.kind() {
            RegionKind::ReLateParam(_)
            | RegionKind::ReErased
            | RegionKind::ReStatic
            | RegionKind::ReEarlyParam(..)
            | RegionKind::ReError(..) => r,

            RegionKind::RePlaceholder(placeholder) => canonicalizer
                .canonical_var_for_region(CanonicalVarKind::PlaceholderRegion(placeholder), r),

            RegionKind::ReVar(vid) => {
                let universe = infcx
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .probe_value(vid)
                    .unwrap_err();
                canonicalizer.canonical_var_for_region(CanonicalVarKind::Region(universe), r)
            }

            _ => {
                // Other than `'static` or `'empty`, the query
                // response should be executing in a fully
                // canonicalized environment, so there shouldn't be
                // any other region names it can come up.
                //
                // rust-lang/rust#57464: `impl Trait` can leak local
                // scopes (in manner violating typeck). Therefore, use
                // `delayed_bug` to allow type error over an ICE.
                panic!("unexpected region in query response: `{r:?}`");
            }
        }
    }

    fn any(&self) -> bool {
        false
    }

    fn preserve_universes(&self) -> bool {
        true
    }
}

struct CanonicalizeUserTypeAnnotation;

impl CanonicalizeMode for CanonicalizeUserTypeAnnotation {
    fn canonicalize_free_region<'db>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'db>,
        r: Region<'db>,
    ) -> Region<'db> {
        match r.kind() {
            RegionKind::ReEarlyParam(_)
            | RegionKind::ReLateParam(_)
            | RegionKind::ReErased
            | RegionKind::ReStatic
            | RegionKind::ReError(_) => r,
            RegionKind::ReVar(_) => canonicalizer.canonical_var_for_region_in_root_universe(r),
            RegionKind::RePlaceholder(..) | RegionKind::ReBound(..) => {
                // We only expect region names that the user can type.
                panic!("unexpected region in query response: `{r:?}`")
            }
        }
    }

    fn any(&self) -> bool {
        false
    }

    fn preserve_universes(&self) -> bool {
        false
    }
}

struct CanonicalizeAllFreeRegions;

impl CanonicalizeMode for CanonicalizeAllFreeRegions {
    fn canonicalize_free_region<'db>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'db>,
        r: Region<'db>,
    ) -> Region<'db> {
        canonicalizer.canonical_var_for_region_in_root_universe(r)
    }

    fn any(&self) -> bool {
        true
    }

    fn preserve_universes(&self) -> bool {
        false
    }
}

struct CanonicalizeFreeRegionsOtherThanStatic;

impl CanonicalizeMode for CanonicalizeFreeRegionsOtherThanStatic {
    fn canonicalize_free_region<'db>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'db>,
        r: Region<'db>,
    ) -> Region<'db> {
        if r.is_static() { r } else { canonicalizer.canonical_var_for_region_in_root_universe(r) }
    }

    fn any(&self) -> bool {
        true
    }

    fn preserve_universes(&self) -> bool {
        false
    }
}

struct Canonicalizer<'cx, 'db> {
    /// Set to `None` to disable the resolution of inference variables.
    infcx: &'cx InferCtxt<'db>,
    tcx: DbInterner<'db>,
    variables: SmallVec<[CanonicalVarKind<'db>; 8]>,
    query_state: &'cx mut OriginalQueryValues<'db>,
    // Note that indices is only used once `var_values` is big enough to be
    // heap-allocated.
    indices: FxHashMap<GenericArg<'db>, BoundVar>,
    /// Maps each `sub_unification_table_root_var` to the index of the first
    /// variable which used it.
    ///
    /// This means in case two type variables have the same sub relations root,
    /// we set the `sub_root` of the second variable to the position of the first.
    /// Otherwise the `sub_root` of each type variable is just its own position.
    sub_root_lookup_table: FxHashMap<TyVid, usize>,
    canonicalize_mode: &'cx dyn CanonicalizeMode,
    needs_canonical_flags: TypeFlags,

    binder_index: DebruijnIndex,
}

impl<'cx, 'db> TypeFolder<DbInterner<'db>> for Canonicalizer<'cx, 'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.tcx
    }

    fn fold_binder<T>(&mut self, t: Binder<'db, T>) -> Binder<'db, T>
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.binder_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        match r.kind() {
            RegionKind::ReBound(index, ..) => {
                if index >= self.binder_index {
                    panic!("escaping late-bound region during canonicalization");
                } else {
                    r
                }
            }

            RegionKind::ReStatic
            | RegionKind::ReEarlyParam(..)
            | RegionKind::ReError(_)
            | RegionKind::ReLateParam(_)
            | RegionKind::RePlaceholder(..)
            | RegionKind::ReVar(_)
            | RegionKind::ReErased => self.canonicalize_mode.canonicalize_free_region(self, r),
        }
    }

    fn fold_ty(&mut self, mut t: Ty<'db>) -> Ty<'db> {
        match t.kind() {
            TyKind::Infer(TyVar(mut vid)) => {
                // We need to canonicalize the *root* of our ty var.
                // This is so that our canonical response correctly reflects
                // any equated inference vars correctly!
                let root_vid = self.infcx.root_var(vid);
                if root_vid != vid {
                    t = Ty::new_var(self.tcx, root_vid);
                    vid = root_vid;
                }

                debug!("canonical: type var found with vid {:?}", vid);
                match self.infcx.probe_ty_var(vid) {
                    // `t` could be a float / int variable; canonicalize that instead.
                    Ok(t) => {
                        debug!("(resolved to {:?})", t);
                        self.fold_ty(t)
                    }

                    // `TyVar(vid)` is unresolved, track its universe index in the canonicalized
                    // result.
                    Err(mut ui) => {
                        if !self.canonicalize_mode.preserve_universes() {
                            // FIXME: perf problem described in #55921.
                            ui = UniverseIndex::ROOT;
                        }

                        let sub_root = self.get_or_insert_sub_root(vid);
                        self.canonicalize_ty_var(CanonicalVarKind::Ty { ui, sub_root }, t)
                    }
                }
            }

            TyKind::Infer(IntVar(vid)) => {
                let nt = self.infcx.opportunistic_resolve_int_var(vid);
                if nt != t {
                    self.fold_ty(nt)
                } else {
                    self.canonicalize_ty_var(CanonicalVarKind::Int, t)
                }
            }
            TyKind::Infer(FloatVar(vid)) => {
                let nt = self.infcx.opportunistic_resolve_float_var(vid);
                if nt != t {
                    self.fold_ty(nt)
                } else {
                    self.canonicalize_ty_var(CanonicalVarKind::Float, t)
                }
            }

            TyKind::Infer(
                InferTy::FreshTy(_) | InferTy::FreshIntTy(_) | InferTy::FreshFloatTy(_),
            ) => {
                panic!("encountered a fresh type during canonicalization")
            }

            TyKind::Placeholder(mut placeholder) => {
                if !self.canonicalize_mode.preserve_universes() {
                    placeholder.universe = UniverseIndex::ROOT;
                }
                self.canonicalize_ty_var(CanonicalVarKind::PlaceholderTy(placeholder), t)
            }

            TyKind::Bound(debruijn, _) => {
                if debruijn >= self.binder_index {
                    panic!("escaping bound type during canonicalization")
                } else {
                    t
                }
            }

            TyKind::Closure(..)
            | TyKind::CoroutineClosure(..)
            | TyKind::Coroutine(..)
            | TyKind::CoroutineWitness(..)
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Int(..)
            | TyKind::Uint(..)
            | TyKind::Float(..)
            | TyKind::Adt(..)
            | TyKind::Str
            | TyKind::Error(_)
            | TyKind::Array(..)
            | TyKind::Slice(..)
            | TyKind::RawPtr(..)
            | TyKind::Ref(..)
            | TyKind::FnDef(..)
            | TyKind::FnPtr(..)
            | TyKind::Dynamic(..)
            | TyKind::UnsafeBinder(_)
            | TyKind::Never
            | TyKind::Tuple(..)
            | TyKind::Alias(..)
            | TyKind::Foreign(..)
            | TyKind::Pat(..)
            | TyKind::Param(..) => {
                if t.flags().intersects(self.needs_canonical_flags) {
                    t.super_fold_with(self)
                } else {
                    t
                }
            }
        }
    }

    fn fold_const(&mut self, mut ct: Const<'db>) -> Const<'db> {
        match ct.kind() {
            ConstKind::Infer(InferConst::Var(mut vid)) => {
                // We need to canonicalize the *root* of our const var.
                // This is so that our canonical response correctly reflects
                // any equated inference vars correctly!
                let root_vid = self.infcx.root_const_var(vid);
                if root_vid != vid {
                    ct = Const::new_var(self.tcx, root_vid);
                    vid = root_vid;
                }

                debug!("canonical: const var found with vid {:?}", vid);
                match self.infcx.probe_const_var(vid) {
                    Ok(c) => {
                        debug!("(resolved to {:?})", c);
                        return self.fold_const(c);
                    }

                    // `ConstVar(vid)` is unresolved, track its universe index in the
                    // canonicalized result
                    Err(mut ui) => {
                        if !self.canonicalize_mode.preserve_universes() {
                            // FIXME: perf problem described in #55921.
                            ui = UniverseIndex::ROOT;
                        }
                        return self.canonicalize_const_var(CanonicalVarKind::Const(ui), ct);
                    }
                }
            }
            ConstKind::Infer(InferConst::Fresh(_)) => {
                panic!("encountered a fresh const during canonicalization")
            }
            ConstKind::Bound(debruijn, _) => {
                if debruijn >= self.binder_index {
                    panic!("escaping bound const during canonicalization")
                } else {
                    return ct;
                }
            }
            ConstKind::Placeholder(placeholder) => {
                return self
                    .canonicalize_const_var(CanonicalVarKind::PlaceholderConst(placeholder), ct);
            }
            _ => {}
        }

        if ct.flags().intersects(self.needs_canonical_flags) {
            ct.super_fold_with(self)
        } else {
            ct
        }
    }
}

impl<'cx, 'db> Canonicalizer<'cx, 'db> {
    /// The main `canonicalize` method, shared impl of
    /// `canonicalize_query` and `canonicalize_response`.
    fn canonicalize<V>(
        value: V,
        infcx: &InferCtxt<'db>,
        tcx: DbInterner<'db>,
        canonicalize_region_mode: &dyn CanonicalizeMode,
        query_state: &mut OriginalQueryValues<'db>,
    ) -> Canonical<'db, V>
    where
        V: TypeFoldable<DbInterner<'db>>,
    {
        let base = Canonical {
            max_universe: UniverseIndex::ROOT,
            variables: CanonicalVars::new_from_iter(tcx, []),
            value: (),
        };
        Canonicalizer::canonicalize_with_base(
            base,
            value,
            infcx,
            tcx,
            canonicalize_region_mode,
            query_state,
        )
        .unchecked_map(|((), val)| val)
    }

    fn canonicalize_with_base<U, V>(
        base: Canonical<'db, U>,
        value: V,
        infcx: &InferCtxt<'db>,
        tcx: DbInterner<'db>,
        canonicalize_region_mode: &dyn CanonicalizeMode,
        query_state: &mut OriginalQueryValues<'db>,
    ) -> Canonical<'db, (U, V)>
    where
        V: TypeFoldable<DbInterner<'db>>,
    {
        let needs_canonical_flags = if canonicalize_region_mode.any() {
            TypeFlags::HAS_INFER | TypeFlags::HAS_PLACEHOLDER | TypeFlags::HAS_FREE_REGIONS
        } else {
            TypeFlags::HAS_INFER | TypeFlags::HAS_PLACEHOLDER
        };

        // Fast path: nothing that needs to be canonicalized.
        if !value.has_type_flags(needs_canonical_flags) {
            return base.unchecked_map(|b| (b, value));
        }

        let mut canonicalizer = Canonicalizer {
            infcx,
            tcx,
            canonicalize_mode: canonicalize_region_mode,
            needs_canonical_flags,
            variables: SmallVec::from_slice(base.variables.as_slice()),
            query_state,
            indices: FxHashMap::default(),
            sub_root_lookup_table: Default::default(),
            binder_index: DebruijnIndex::ZERO,
        };
        if canonicalizer.query_state.var_values.spilled() {
            canonicalizer.indices = canonicalizer
                .query_state
                .var_values
                .iter()
                .enumerate()
                .map(|(i, &kind)| (kind, BoundVar::from(i)))
                .collect();
        }
        let out_value = value.fold_with(&mut canonicalizer);

        // Once we have canonicalized `out_value`, it should not
        // contain anything that ties it to this inference context
        // anymore.
        debug_assert!(!out_value.has_infer() && !out_value.has_placeholders());

        let canonical_variables =
            CanonicalVars::new_from_iter(tcx, canonicalizer.universe_canonicalized_variables());

        let max_universe = canonical_variables
            .iter()
            .map(|cvar| cvar.universe())
            .max()
            .unwrap_or(UniverseIndex::ROOT);

        Canonical { max_universe, variables: canonical_variables, value: (base.value, out_value) }
    }

    /// Creates a canonical variable replacing `kind` from the input,
    /// or returns an existing variable if `kind` has already been
    /// seen. `kind` is expected to be an unbound variable (or
    /// potentially a free region).
    fn canonical_var(&mut self, info: CanonicalVarKind<'db>, kind: GenericArg<'db>) -> BoundVar {
        let Canonicalizer { variables, query_state, indices, .. } = self;

        let var_values = &mut query_state.var_values;

        let universe = info.universe();
        if universe != UniverseIndex::ROOT {
            assert!(self.canonicalize_mode.preserve_universes());

            // Insert universe into the universe map. To preserve the order of the
            // universes in the value being canonicalized, we don't update the
            // universe in `info` until we have finished canonicalizing.
            match query_state.universe_map.binary_search(&universe) {
                Err(idx) => query_state.universe_map.insert(idx, universe),
                Ok(_) => {}
            }
        }

        // This code is hot. `variables` and `var_values` are usually small
        // (fewer than 8 elements ~95% of the time). They are SmallVec's to
        // avoid allocations in those cases. We also don't use `indices` to
        // determine if a kind has been seen before until the limit of 8 has
        // been exceeded, to also avoid allocations for `indices`.
        if !var_values.spilled() {
            // `var_values` is stack-allocated. `indices` isn't used yet. Do a
            // direct linear search of `var_values`.
            if let Some(idx) = var_values.iter().position(|&k| k == kind) {
                // `kind` is already present in `var_values`.
                BoundVar::new(idx)
            } else {
                // `kind` isn't present in `var_values`. Append it. Likewise
                // for `info` and `variables`.
                variables.push(info);
                var_values.push(kind);
                assert_eq!(variables.len(), var_values.len());

                // If `var_values` has become big enough to be heap-allocated,
                // fill up `indices` to facilitate subsequent lookups.
                if var_values.spilled() {
                    assert!(indices.is_empty());
                    *indices = var_values
                        .iter()
                        .enumerate()
                        .map(|(i, &kind)| (kind, BoundVar::new(i)))
                        .collect();
                }
                // The cv is the index of the appended element.
                BoundVar::new(var_values.len() - 1)
            }
        } else {
            // `var_values` is large. Do a hashmap search via `indices`.
            *indices.entry(kind).or_insert_with(|| {
                variables.push(info);
                var_values.push(kind);
                assert_eq!(variables.len(), var_values.len());
                BoundVar::new(variables.len() - 1)
            })
        }
    }

    fn get_or_insert_sub_root(&mut self, vid: TyVid) -> BoundVar {
        let root_vid = self.infcx.sub_unification_table_root_var(vid);
        let idx =
            *self.sub_root_lookup_table.entry(root_vid).or_insert_with(|| self.variables.len());
        BoundVar::from(idx)
    }

    /// Replaces the universe indexes used in `var_values` with their index in
    /// `query_state.universe_map`. This minimizes the maximum universe used in
    /// the canonicalized value.
    fn universe_canonicalized_variables(self) -> SmallVec<[CanonicalVarKind<'db>; 8]> {
        if self.query_state.universe_map.len() == 1 {
            return self.variables;
        }

        let reverse_universe_map: FxHashMap<UniverseIndex, UniverseIndex> = self
            .query_state
            .universe_map
            .iter()
            .enumerate()
            .map(|(idx, universe)| (*universe, UniverseIndex::from_usize(idx)))
            .collect();

        self.variables
            .iter()
            .map(|v| match *v {
                CanonicalVarKind::Int | CanonicalVarKind::Float => *v,
                CanonicalVarKind::Ty { ui, sub_root } => {
                    CanonicalVarKind::Ty { ui: reverse_universe_map[&ui], sub_root }
                }
                CanonicalVarKind::Region(u) => CanonicalVarKind::Region(reverse_universe_map[&u]),
                CanonicalVarKind::Const(u) => CanonicalVarKind::Const(reverse_universe_map[&u]),
                CanonicalVarKind::PlaceholderTy(placeholder) => {
                    CanonicalVarKind::PlaceholderTy(Placeholder {
                        universe: reverse_universe_map[&placeholder.universe],
                        ..placeholder
                    })
                }
                CanonicalVarKind::PlaceholderRegion(placeholder) => {
                    CanonicalVarKind::PlaceholderRegion(Placeholder {
                        universe: reverse_universe_map[&placeholder.universe],
                        ..placeholder
                    })
                }
                CanonicalVarKind::PlaceholderConst(placeholder) => {
                    CanonicalVarKind::PlaceholderConst(Placeholder {
                        universe: reverse_universe_map[&placeholder.universe],
                        ..placeholder
                    })
                }
            })
            .collect()
    }

    /// Shorthand helper that creates a canonical region variable for
    /// `r` (always in the root universe). The reason that we always
    /// put these variables into the root universe is because this
    /// method is used during **query construction:** in that case, we
    /// are taking all the regions and just putting them into the most
    /// generic context we can. This may generate solutions that don't
    /// fit (e.g., that equate some region variable with a placeholder
    /// it can't name) on the caller side, but that's ok, the caller
    /// can figure that out. In the meantime, it maximizes our
    /// caching.
    ///
    /// (This works because unification never fails -- and hence trait
    /// selection is never affected -- due to a universe mismatch.)
    fn canonical_var_for_region_in_root_universe(&mut self, r: Region<'db>) -> Region<'db> {
        self.canonical_var_for_region(CanonicalVarKind::Region(UniverseIndex::ROOT), r)
    }

    /// Creates a canonical variable (with the given `info`)
    /// representing the region `r`; return a region referencing it.
    fn canonical_var_for_region(
        &mut self,
        info: CanonicalVarKind<'db>,
        r: Region<'db>,
    ) -> Region<'db> {
        let var = self.canonical_var(info, r.into());
        let br = BoundRegion { var, kind: BoundRegionKind::Anon };
        Region::new_bound(self.cx(), self.binder_index, br)
    }

    /// Given a type variable `ty_var` of the given kind, first check
    /// if `ty_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `ty_var`.
    fn canonicalize_ty_var(&mut self, info: CanonicalVarKind<'db>, ty_var: Ty<'db>) -> Ty<'db> {
        debug_assert_eq!(ty_var, self.infcx.shallow_resolve(ty_var));
        let var = self.canonical_var(info, ty_var.into());
        Ty::new_bound(
            self.tcx,
            self.binder_index,
            BoundTy { kind: crate::next_solver::BoundTyKind::Anon, var },
        )
    }

    /// Given a type variable `const_var` of the given kind, first check
    /// if `const_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `const_var`.
    fn canonicalize_const_var(
        &mut self,
        info: CanonicalVarKind<'db>,
        const_var: Const<'db>,
    ) -> Const<'db> {
        debug_assert_eq!(const_var, self.infcx.shallow_resolve_const(const_var));
        let var = self.canonical_var(info, const_var.into());
        Const::new_bound(self.tcx, self.binder_index, BoundConst { var })
    }
}
