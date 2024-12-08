//! This module contains the "canonicalizer" itself.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use rustc_data_structures::fx::FxHashMap;
use rustc_index::Idx;
use rustc_middle::bug;
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::{
    self, BoundVar, GenericArg, InferConst, List, Ty, TyCtxt, TypeFlags, TypeVisitableExt,
};
use smallvec::SmallVec;
use tracing::debug;

use crate::infer::InferCtxt;
use crate::infer::canonical::{
    Canonical, CanonicalQueryInput, CanonicalTyVarKind, CanonicalVarInfo, CanonicalVarKind,
    OriginalQueryValues,
};

impl<'tcx> InferCtxt<'tcx> {
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
        value: ty::ParamEnvAnd<'tcx, V>,
        query_state: &mut OriginalQueryValues<'tcx>,
    ) -> CanonicalQueryInput<'tcx, ty::ParamEnvAnd<'tcx, V>>
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
    {
        let (param_env, value) = value.into_parts();
        let canonical_param_env = self.tcx.canonical_param_env_cache.get_or_insert(
            self.tcx,
            param_env,
            query_state,
            |tcx, param_env, query_state| {
                // FIXME(#118965): We don't canonicalize the static lifetimes that appear in the
                // `param_env` because they are treated differently by trait selection.
                Canonicalizer::canonicalize(
                    param_env,
                    None,
                    tcx,
                    &CanonicalizeFreeRegionsOtherThanStatic,
                    query_state,
                )
            },
        );

        let canonical = Canonicalizer::canonicalize_with_base(
            canonical_param_env,
            value,
            Some(self),
            self.tcx,
            &CanonicalizeAllFreeRegions,
            query_state,
        )
        .unchecked_map(|(param_env, value)| param_env.and(value));
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
    pub fn canonicalize_response<V>(&self, value: V) -> Canonical<'tcx, V>
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut query_state = OriginalQueryValues::default();
        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            &CanonicalizeQueryResponse,
            &mut query_state,
        )
    }

    pub fn canonicalize_user_type_annotation<V>(&self, value: V) -> Canonical<'tcx, V>
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut query_state = OriginalQueryValues::default();
        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
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
    fn canonicalize_free_region<'tcx>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx>;

    fn any(&self) -> bool;

    // Do we preserve universe of variables.
    fn preserve_universes(&self) -> bool;
}

struct CanonicalizeQueryResponse;

impl CanonicalizeMode for CanonicalizeQueryResponse {
    fn canonicalize_free_region<'tcx>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        mut r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        let infcx = canonicalizer.infcx.unwrap();

        if let ty::ReVar(vid) = *r {
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

        match *r {
            ty::ReLateParam(_) | ty::ReErased | ty::ReStatic | ty::ReEarlyParam(..) => r,

            ty::RePlaceholder(placeholder) => canonicalizer.canonical_var_for_region(
                CanonicalVarInfo { kind: CanonicalVarKind::PlaceholderRegion(placeholder) },
                r,
            ),

            ty::ReVar(vid) => {
                let universe = infcx
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .probe_value(vid)
                    .unwrap_err();
                canonicalizer.canonical_var_for_region(
                    CanonicalVarInfo { kind: CanonicalVarKind::Region(universe) },
                    r,
                )
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
                canonicalizer
                    .tcx
                    .dcx()
                    .delayed_bug(format!("unexpected region in query response: `{r:?}`"));
                r
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
    fn canonicalize_free_region<'tcx>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        match *r {
            ty::ReEarlyParam(_)
            | ty::ReLateParam(_)
            | ty::ReErased
            | ty::ReStatic
            | ty::ReError(_) => r,
            ty::ReVar(_) => canonicalizer.canonical_var_for_region_in_root_universe(r),
            ty::RePlaceholder(..) | ty::ReBound(..) => {
                // We only expect region names that the user can type.
                bug!("unexpected region in query response: `{:?}`", r)
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
    fn canonicalize_free_region<'tcx>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
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
    fn canonicalize_free_region<'tcx>(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        if r.is_static() { r } else { canonicalizer.canonical_var_for_region_in_root_universe(r) }
    }

    fn any(&self) -> bool {
        true
    }

    fn preserve_universes(&self) -> bool {
        false
    }
}

struct Canonicalizer<'cx, 'tcx> {
    /// Set to `None` to disable the resolution of inference variables.
    infcx: Option<&'cx InferCtxt<'tcx>>,
    tcx: TyCtxt<'tcx>,
    variables: SmallVec<[CanonicalVarInfo<'tcx>; 8]>,
    query_state: &'cx mut OriginalQueryValues<'tcx>,
    // Note that indices is only used once `var_values` is big enough to be
    // heap-allocated.
    indices: FxHashMap<GenericArg<'tcx>, BoundVar>,
    canonicalize_mode: &'cx dyn CanonicalizeMode,
    needs_canonical_flags: TypeFlags,

    binder_index: ty::DebruijnIndex,
}

impl<'cx, 'tcx> TypeFolder<TyCtxt<'tcx>> for Canonicalizer<'cx, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.binder_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReBound(index, ..) => {
                if index >= self.binder_index {
                    bug!("escaping late-bound region during canonicalization");
                } else {
                    r
                }
            }

            ty::ReStatic
            | ty::ReEarlyParam(..)
            | ty::ReError(_)
            | ty::ReLateParam(_)
            | ty::RePlaceholder(..)
            | ty::ReVar(_)
            | ty::ReErased => self.canonicalize_mode.canonicalize_free_region(self, r),
        }
    }

    fn fold_ty(&mut self, mut t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Infer(ty::TyVar(mut vid)) => {
                // We need to canonicalize the *root* of our ty var.
                // This is so that our canonical response correctly reflects
                // any equated inference vars correctly!
                let root_vid = self.infcx.unwrap().root_var(vid);
                if root_vid != vid {
                    t = Ty::new_var(self.tcx, root_vid);
                    vid = root_vid;
                }

                debug!("canonical: type var found with vid {:?}", vid);
                match self.infcx.unwrap().probe_ty_var(vid) {
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
                            ui = ty::UniverseIndex::ROOT;
                        }
                        self.canonicalize_ty_var(
                            CanonicalVarInfo {
                                kind: CanonicalVarKind::Ty(CanonicalTyVarKind::General(ui)),
                            },
                            t,
                        )
                    }
                }
            }

            ty::Infer(ty::IntVar(vid)) => {
                let nt = self.infcx.unwrap().opportunistic_resolve_int_var(vid);
                if nt != t {
                    return self.fold_ty(nt);
                } else {
                    self.canonicalize_ty_var(
                        CanonicalVarInfo { kind: CanonicalVarKind::Ty(CanonicalTyVarKind::Int) },
                        t,
                    )
                }
            }
            ty::Infer(ty::FloatVar(vid)) => {
                let nt = self.infcx.unwrap().opportunistic_resolve_float_var(vid);
                if nt != t {
                    return self.fold_ty(nt);
                } else {
                    self.canonicalize_ty_var(
                        CanonicalVarInfo { kind: CanonicalVarKind::Ty(CanonicalTyVarKind::Float) },
                        t,
                    )
                }
            }

            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("encountered a fresh type during canonicalization")
            }

            ty::Placeholder(mut placeholder) => {
                if !self.canonicalize_mode.preserve_universes() {
                    placeholder.universe = ty::UniverseIndex::ROOT;
                }
                self.canonicalize_ty_var(
                    CanonicalVarInfo { kind: CanonicalVarKind::PlaceholderTy(placeholder) },
                    t,
                )
            }

            ty::Bound(debruijn, _) => {
                if debruijn >= self.binder_index {
                    bug!("escaping bound type during canonicalization")
                } else {
                    t
                }
            }

            ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Adt(..)
            | ty::Str
            | ty::Error(_)
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Dynamic(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::Alias(..)
            | ty::Foreign(..)
            | ty::Pat(..)
            | ty::Param(..) => {
                if t.flags().intersects(self.needs_canonical_flags) {
                    t.super_fold_with(self)
                } else {
                    t
                }
            }
        }
    }

    fn fold_const(&mut self, mut ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Infer(InferConst::Var(mut vid)) => {
                // We need to canonicalize the *root* of our const var.
                // This is so that our canonical response correctly reflects
                // any equated inference vars correctly!
                let root_vid = self.infcx.unwrap().root_const_var(vid);
                if root_vid != vid {
                    ct = ty::Const::new_var(self.tcx, root_vid);
                    vid = root_vid;
                }

                debug!("canonical: const var found with vid {:?}", vid);
                match self.infcx.unwrap().probe_const_var(vid) {
                    Ok(c) => {
                        debug!("(resolved to {:?})", c);
                        return self.fold_const(c);
                    }

                    // `ConstVar(vid)` is unresolved, track its universe index in the
                    // canonicalized result
                    Err(mut ui) => {
                        if !self.canonicalize_mode.preserve_universes() {
                            // FIXME: perf problem described in #55921.
                            ui = ty::UniverseIndex::ROOT;
                        }
                        return self.canonicalize_const_var(
                            CanonicalVarInfo { kind: CanonicalVarKind::Const(ui) },
                            ct,
                        );
                    }
                }
            }
            ty::ConstKind::Infer(InferConst::Fresh(_)) => {
                bug!("encountered a fresh const during canonicalization")
            }
            ty::ConstKind::Bound(debruijn, _) => {
                if debruijn >= self.binder_index {
                    bug!("escaping bound const during canonicalization")
                } else {
                    return ct;
                }
            }
            ty::ConstKind::Placeholder(placeholder) => {
                return self.canonicalize_const_var(
                    CanonicalVarInfo { kind: CanonicalVarKind::PlaceholderConst(placeholder) },
                    ct,
                );
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

impl<'cx, 'tcx> Canonicalizer<'cx, 'tcx> {
    /// The main `canonicalize` method, shared impl of
    /// `canonicalize_query` and `canonicalize_response`.
    fn canonicalize<V>(
        value: V,
        infcx: Option<&InferCtxt<'tcx>>,
        tcx: TyCtxt<'tcx>,
        canonicalize_region_mode: &dyn CanonicalizeMode,
        query_state: &mut OriginalQueryValues<'tcx>,
    ) -> Canonical<'tcx, V>
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
    {
        let base = Canonical {
            max_universe: ty::UniverseIndex::ROOT,
            variables: List::empty(),
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
        base: Canonical<'tcx, U>,
        value: V,
        infcx: Option<&InferCtxt<'tcx>>,
        tcx: TyCtxt<'tcx>,
        canonicalize_region_mode: &dyn CanonicalizeMode,
        query_state: &mut OriginalQueryValues<'tcx>,
    ) -> Canonical<'tcx, (U, V)>
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
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
            variables: SmallVec::from_slice(base.variables),
            query_state,
            indices: FxHashMap::default(),
            binder_index: ty::INNERMOST,
        };
        if canonicalizer.query_state.var_values.spilled() {
            canonicalizer.indices = canonicalizer
                .query_state
                .var_values
                .iter()
                .enumerate()
                .map(|(i, &kind)| (kind, BoundVar::new(i)))
                .collect();
        }
        let out_value = value.fold_with(&mut canonicalizer);

        // Once we have canonicalized `out_value`, it should not
        // contain anything that ties it to this inference context
        // anymore.
        debug_assert!(!out_value.has_infer() && !out_value.has_placeholders());

        let canonical_variables =
            tcx.mk_canonical_var_infos(&canonicalizer.universe_canonicalized_variables());

        let max_universe = canonical_variables
            .iter()
            .map(|cvar| cvar.universe())
            .max()
            .unwrap_or(ty::UniverseIndex::ROOT);

        Canonical { max_universe, variables: canonical_variables, value: (base.value, out_value) }
    }

    /// Creates a canonical variable replacing `kind` from the input,
    /// or returns an existing variable if `kind` has already been
    /// seen. `kind` is expected to be an unbound variable (or
    /// potentially a free region).
    fn canonical_var(&mut self, info: CanonicalVarInfo<'tcx>, kind: GenericArg<'tcx>) -> BoundVar {
        let Canonicalizer { variables, query_state, indices, .. } = self;

        let var_values = &mut query_state.var_values;

        let universe = info.universe();
        if universe != ty::UniverseIndex::ROOT {
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

    /// Replaces the universe indexes used in `var_values` with their index in
    /// `query_state.universe_map`. This minimizes the maximum universe used in
    /// the canonicalized value.
    fn universe_canonicalized_variables(self) -> SmallVec<[CanonicalVarInfo<'tcx>; 8]> {
        if self.query_state.universe_map.len() == 1 {
            return self.variables;
        }

        let reverse_universe_map: FxHashMap<ty::UniverseIndex, ty::UniverseIndex> = self
            .query_state
            .universe_map
            .iter()
            .enumerate()
            .map(|(idx, universe)| (*universe, ty::UniverseIndex::from_usize(idx)))
            .collect();

        self.variables
            .iter()
            .map(|v| CanonicalVarInfo {
                kind: match v.kind {
                    CanonicalVarKind::Ty(CanonicalTyVarKind::Int | CanonicalTyVarKind::Float) => {
                        return *v;
                    }
                    CanonicalVarKind::Ty(CanonicalTyVarKind::General(u)) => {
                        CanonicalVarKind::Ty(CanonicalTyVarKind::General(reverse_universe_map[&u]))
                    }
                    CanonicalVarKind::Region(u) => {
                        CanonicalVarKind::Region(reverse_universe_map[&u])
                    }
                    CanonicalVarKind::Const(u) => CanonicalVarKind::Const(reverse_universe_map[&u]),
                    CanonicalVarKind::PlaceholderTy(placeholder) => {
                        CanonicalVarKind::PlaceholderTy(ty::Placeholder {
                            universe: reverse_universe_map[&placeholder.universe],
                            ..placeholder
                        })
                    }
                    CanonicalVarKind::PlaceholderRegion(placeholder) => {
                        CanonicalVarKind::PlaceholderRegion(ty::Placeholder {
                            universe: reverse_universe_map[&placeholder.universe],
                            ..placeholder
                        })
                    }
                    CanonicalVarKind::PlaceholderConst(placeholder) => {
                        CanonicalVarKind::PlaceholderConst(ty::Placeholder {
                            universe: reverse_universe_map[&placeholder.universe],
                            ..placeholder
                        })
                    }
                },
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
    fn canonical_var_for_region_in_root_universe(
        &mut self,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        self.canonical_var_for_region(
            CanonicalVarInfo { kind: CanonicalVarKind::Region(ty::UniverseIndex::ROOT) },
            r,
        )
    }

    /// Creates a canonical variable (with the given `info`)
    /// representing the region `r`; return a region referencing it.
    fn canonical_var_for_region(
        &mut self,
        info: CanonicalVarInfo<'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        let var = self.canonical_var(info, r.into());
        let br = ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon };
        ty::Region::new_bound(self.cx(), self.binder_index, br)
    }

    /// Given a type variable `ty_var` of the given kind, first check
    /// if `ty_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `ty_var`.
    fn canonicalize_ty_var(&mut self, info: CanonicalVarInfo<'tcx>, ty_var: Ty<'tcx>) -> Ty<'tcx> {
        debug_assert!(!self.infcx.is_some_and(|infcx| ty_var != infcx.shallow_resolve(ty_var)));
        let var = self.canonical_var(info, ty_var.into());
        Ty::new_bound(self.tcx, self.binder_index, var.into())
    }

    /// Given a type variable `const_var` of the given kind, first check
    /// if `const_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `const_var`.
    fn canonicalize_const_var(
        &mut self,
        info: CanonicalVarInfo<'tcx>,
        const_var: ty::Const<'tcx>,
    ) -> ty::Const<'tcx> {
        debug_assert!(
            !self.infcx.is_some_and(|infcx| const_var != infcx.shallow_resolve_const(const_var))
        );
        let var = self.canonical_var(info, const_var.into());
        ty::Const::new_bound(self.tcx, self.binder_index, var)
    }
}
