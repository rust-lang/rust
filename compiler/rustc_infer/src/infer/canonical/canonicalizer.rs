//! This module contains the "canonicalizer" itself.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use crate::infer::canonical::{
    Canonical, CanonicalTyVarKind, CanonicalVarInfo, CanonicalVarKind, Canonicalized,
    OriginalQueryValues,
};
use crate::infer::InferCtxt;
use rustc_middle::ty::flags::FlagComputation;
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder};
use rustc_middle::ty::subst::GenericArg;
use rustc_middle::ty::{self, BoundVar, InferConst, List, Ty, TyCtxt, TypeFlags};
use std::sync::atomic::Ordering;

use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::Idx;
use smallvec::SmallVec;

impl<'cx, 'tcx> InferCtxt<'cx, 'tcx> {
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
        value: V,
        query_state: &mut OriginalQueryValues<'tcx>,
    ) -> Canonicalized<'tcx, V>
    where
        V: TypeFoldable<'tcx>,
    {
        self.tcx.sess.perf_stats.queries_canonicalized.fetch_add(1, Ordering::Relaxed);

        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            &CanonicalizeAllFreeRegions,
            query_state,
        )
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
    ///
    ///     impl<T> Trait<'static> for T { .. }
    ///
    /// then we would prepare a query result R that (among other
    /// things) includes a mapping to `'?0 := 'static`. When
    /// canonicalizing this query result R, we would leave this
    /// reference to `'static` alone.
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc dev guide][c].
    ///
    /// [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html#canonicalizing-the-query-result
    pub fn canonicalize_response<V>(&self, value: V) -> Canonicalized<'tcx, V>
    where
        V: TypeFoldable<'tcx>,
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

    pub fn canonicalize_user_type_annotation<V>(&self, value: V) -> Canonicalized<'tcx, V>
    where
        V: TypeFoldable<'tcx>,
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

    /// A hacky variant of `canonicalize_query` that does not
    /// canonicalize `'static`. Unfortunately, the existing leak
    /// check treats `'static` differently in some cases (see also
    /// #33684), so if we are performing an operation that may need to
    /// prove "leak-check" related things, we leave `'static`
    /// alone.
    ///
    /// `'static` is also special cased when winnowing candidates when
    /// selecting implementation candidates, so we also have to leave `'static`
    /// alone for queries that do selection.
    //
    // FIXME(#48536): once the above issues are resolved, we can remove this
    // and just use `canonicalize_query`.
    pub fn canonicalize_hr_query_hack<V>(
        &self,
        value: V,
        query_state: &mut OriginalQueryValues<'tcx>,
    ) -> Canonicalized<'tcx, V>
    where
        V: TypeFoldable<'tcx>,
    {
        self.tcx.sess.perf_stats.queries_canonicalized.fetch_add(1, Ordering::Relaxed);

        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            &CanonicalizeFreeRegionsOtherThanStatic,
            query_state,
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
trait CanonicalizeRegionMode {
    fn canonicalize_free_region(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx>;

    fn any(&self) -> bool;
}

struct CanonicalizeQueryResponse;

impl CanonicalizeRegionMode for CanonicalizeQueryResponse {
    fn canonicalize_free_region(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        match r {
            ty::ReFree(_)
            | ty::ReErased
            | ty::ReStatic
            | ty::ReEmpty(ty::UniverseIndex::ROOT)
            | ty::ReEarlyBound(..) => r,

            ty::RePlaceholder(placeholder) => canonicalizer.canonical_var_for_region(
                CanonicalVarInfo { kind: CanonicalVarKind::PlaceholderRegion(*placeholder) },
                r,
            ),

            ty::ReVar(vid) => {
                let universe = canonicalizer.region_var_universe(*vid);
                canonicalizer.canonical_var_for_region(
                    CanonicalVarInfo { kind: CanonicalVarKind::Region(universe) },
                    r,
                )
            }

            ty::ReEmpty(ui) => {
                bug!("canonicalizing 'empty in universe {:?}", ui) // FIXME
            }

            _ => {
                // Other than `'static` or `'empty`, the query
                // response should be executing in a fully
                // canonicalized environment, so there shouldn't be
                // any other region names it can come up.
                //
                // rust-lang/rust#57464: `impl Trait` can leak local
                // scopes (in manner violating typeck). Therefore, use
                // `delay_span_bug` to allow type error over an ICE.
                ty::tls::with(|tcx| {
                    tcx.sess.delay_span_bug(
                        rustc_span::DUMMY_SP,
                        &format!("unexpected region in query response: `{:?}`", r),
                    );
                });
                r
            }
        }
    }

    fn any(&self) -> bool {
        false
    }
}

struct CanonicalizeUserTypeAnnotation;

impl CanonicalizeRegionMode for CanonicalizeUserTypeAnnotation {
    fn canonicalize_free_region(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        match r {
            ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReErased | ty::ReStatic => r,
            ty::ReVar(_) => canonicalizer.canonical_var_for_region_in_root_universe(r),
            _ => {
                // We only expect region names that the user can type.
                bug!("unexpected region in query response: `{:?}`", r)
            }
        }
    }

    fn any(&self) -> bool {
        false
    }
}

struct CanonicalizeAllFreeRegions;

impl CanonicalizeRegionMode for CanonicalizeAllFreeRegions {
    fn canonicalize_free_region(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        canonicalizer.canonical_var_for_region_in_root_universe(r)
    }

    fn any(&self) -> bool {
        true
    }
}

struct CanonicalizeFreeRegionsOtherThanStatic;

impl CanonicalizeRegionMode for CanonicalizeFreeRegionsOtherThanStatic {
    fn canonicalize_free_region(
        &self,
        canonicalizer: &mut Canonicalizer<'_, 'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        if let ty::ReStatic = r {
            r
        } else {
            canonicalizer.canonical_var_for_region_in_root_universe(r)
        }
    }

    fn any(&self) -> bool {
        true
    }
}

struct Canonicalizer<'cx, 'tcx> {
    infcx: Option<&'cx InferCtxt<'cx, 'tcx>>,
    tcx: TyCtxt<'tcx>,
    variables: SmallVec<[CanonicalVarInfo<'tcx>; 8]>,
    query_state: &'cx mut OriginalQueryValues<'tcx>,
    // Note that indices is only used once `var_values` is big enough to be
    // heap-allocated.
    indices: FxHashMap<GenericArg<'tcx>, BoundVar>,
    canonicalize_region_mode: &'cx dyn CanonicalizeRegionMode,
    needs_canonical_flags: TypeFlags,

    binder_index: ty::DebruijnIndex,
}

impl<'cx, 'tcx> TypeFolder<'tcx> for Canonicalizer<'cx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<T>) -> ty::Binder<T>
    where
        T: TypeFoldable<'tcx>,
    {
        self.binder_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(index, ..) => {
                if index >= self.binder_index {
                    bug!("escaping late-bound region during canonicalization");
                } else {
                    r
                }
            }

            ty::ReVar(vid) => {
                let resolved_vid = self
                    .infcx
                    .unwrap()
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .opportunistic_resolve_var(vid);
                debug!(
                    "canonical: region var found with vid {:?}, \
                     opportunistically resolved to {:?}",
                    vid, r
                );
                let r = self.tcx.reuse_or_mk_region(r, ty::ReVar(resolved_vid));
                self.canonicalize_region_mode.canonicalize_free_region(self, r)
            }

            ty::ReStatic
            | ty::ReEarlyBound(..)
            | ty::ReFree(_)
            | ty::ReEmpty(_)
            | ty::RePlaceholder(..)
            | ty::ReErased => self.canonicalize_region_mode.canonicalize_free_region(self, r),
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Infer(ty::TyVar(vid)) => {
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
                        if !self.infcx.unwrap().tcx.sess.opts.debugging_opts.chalk {
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

            ty::Infer(ty::IntVar(_)) => self.canonicalize_ty_var(
                CanonicalVarInfo { kind: CanonicalVarKind::Ty(CanonicalTyVarKind::Int) },
                t,
            ),

            ty::Infer(ty::FloatVar(_)) => self.canonicalize_ty_var(
                CanonicalVarInfo { kind: CanonicalVarKind::Ty(CanonicalTyVarKind::Float) },
                t,
            ),

            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("encountered a fresh type during canonicalization")
            }

            ty::Placeholder(placeholder) => self.canonicalize_ty_var(
                CanonicalVarInfo { kind: CanonicalVarKind::PlaceholderTy(placeholder) },
                t,
            ),

            ty::Bound(debruijn, _) => {
                if debruijn >= self.binder_index {
                    bug!("escaping bound type during canonicalization")
                } else {
                    t
                }
            }

            ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
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
            | ty::FnPtr(_)
            | ty::Dynamic(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::Projection(..)
            | ty::Foreign(..)
            | ty::Param(..)
            | ty::Opaque(..) => {
                if t.flags().intersects(self.needs_canonical_flags) {
                    t.super_fold_with(self)
                } else {
                    t
                }
            }
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        match ct.val {
            ty::ConstKind::Infer(InferConst::Var(vid)) => {
                debug!("canonical: const var found with vid {:?}", vid);
                match self.infcx.unwrap().probe_const_var(vid) {
                    Ok(c) => {
                        debug!("(resolved to {:?})", c);
                        return self.fold_const(c);
                    }

                    // `ConstVar(vid)` is unresolved, track its universe index in the
                    // canonicalized result
                    Err(mut ui) => {
                        if !self.infcx.unwrap().tcx.sess.opts.debugging_opts.chalk {
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
                    bug!("escaping bound type during canonicalization")
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

        let flags = FlagComputation::for_const(ct);
        if flags.intersects(self.needs_canonical_flags) { ct.super_fold_with(self) } else { ct }
    }
}

impl<'cx, 'tcx> Canonicalizer<'cx, 'tcx> {
    /// The main `canonicalize` method, shared impl of
    /// `canonicalize_query` and `canonicalize_response`.
    fn canonicalize<V>(
        value: V,
        infcx: Option<&InferCtxt<'_, 'tcx>>,
        tcx: TyCtxt<'tcx>,
        canonicalize_region_mode: &dyn CanonicalizeRegionMode,
        query_state: &mut OriginalQueryValues<'tcx>,
    ) -> Canonicalized<'tcx, V>
    where
        V: TypeFoldable<'tcx>,
    {
        let needs_canonical_flags = if canonicalize_region_mode.any() {
            TypeFlags::NEEDS_INFER |
            TypeFlags::HAS_FREE_REGIONS | // `HAS_RE_PLACEHOLDER` implies `HAS_FREE_REGIONS`
            TypeFlags::HAS_TY_PLACEHOLDER |
            TypeFlags::HAS_CT_PLACEHOLDER
        } else {
            TypeFlags::NEEDS_INFER
                | TypeFlags::HAS_RE_PLACEHOLDER
                | TypeFlags::HAS_TY_PLACEHOLDER
                | TypeFlags::HAS_CT_PLACEHOLDER
        };

        // Fast path: nothing that needs to be canonicalized.
        if !value.has_type_flags(needs_canonical_flags) {
            let canon_value = Canonical {
                max_universe: ty::UniverseIndex::ROOT,
                variables: List::empty(),
                value,
            };
            return canon_value;
        }

        let mut canonicalizer = Canonicalizer {
            infcx,
            tcx,
            canonicalize_region_mode,
            needs_canonical_flags,
            variables: SmallVec::new(),
            query_state,
            indices: FxHashMap::default(),
            binder_index: ty::INNERMOST,
        };
        let out_value = value.fold_with(&mut canonicalizer);

        // Once we have canonicalized `out_value`, it should not
        // contain anything that ties it to this inference context
        // anymore, so it should live in the global arena.
        debug_assert!(!out_value.needs_infer());

        let canonical_variables = tcx.intern_canonical_var_infos(&canonicalizer.variables);

        let max_universe = canonical_variables
            .iter()
            .map(|cvar| cvar.universe())
            .max()
            .unwrap_or(ty::UniverseIndex::ROOT);

        Canonical { max_universe, variables: canonical_variables, value: out_value }
    }

    /// Creates a canonical variable replacing `kind` from the input,
    /// or returns an existing variable if `kind` has already been
    /// seen. `kind` is expected to be an unbound variable (or
    /// potentially a free region).
    fn canonical_var(&mut self, info: CanonicalVarInfo<'tcx>, kind: GenericArg<'tcx>) -> BoundVar {
        let Canonicalizer { variables, query_state, indices, .. } = self;

        let var_values = &mut query_state.var_values;

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

    /// Returns the universe in which `vid` is defined.
    fn region_var_universe(&self, vid: ty::RegionVid) -> ty::UniverseIndex {
        self.infcx.unwrap().inner.borrow_mut().unwrap_region_constraints().var_universe(vid)
    }

    /// Creates a canonical variable (with the given `info`)
    /// representing the region `r`; return a region referencing it.
    fn canonical_var_for_region(
        &mut self,
        info: CanonicalVarInfo<'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        let var = self.canonical_var(info, r.into());
        let br = ty::BoundRegion { kind: ty::BrAnon(var.as_u32()) };
        let region = ty::ReLateBound(self.binder_index, br);
        self.tcx().mk_region(region)
    }

    /// Given a type variable `ty_var` of the given kind, first check
    /// if `ty_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `ty_var`.
    fn canonicalize_ty_var(&mut self, info: CanonicalVarInfo<'tcx>, ty_var: Ty<'tcx>) -> Ty<'tcx> {
        let infcx = self.infcx.expect("encountered ty-var without infcx");
        let bound_to = infcx.shallow_resolve(ty_var);
        if bound_to != ty_var {
            self.fold_ty(bound_to)
        } else {
            let var = self.canonical_var(info, ty_var.into());
            self.tcx().mk_ty(ty::Bound(self.binder_index, var.into()))
        }
    }

    /// Given a type variable `const_var` of the given kind, first check
    /// if `const_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `const_var`.
    fn canonicalize_const_var(
        &mut self,
        info: CanonicalVarInfo<'tcx>,
        const_var: &'tcx ty::Const<'tcx>,
    ) -> &'tcx ty::Const<'tcx> {
        let infcx = self.infcx.expect("encountered const-var without infcx");
        let bound_to = infcx.shallow_resolve(const_var);
        if bound_to != const_var {
            self.fold_const(bound_to)
        } else {
            let var = self.canonical_var(info, const_var.into());
            self.tcx().mk_const(ty::Const {
                val: ty::ConstKind::Bound(self.binder_index, var),
                ty: self.fold_ty(const_var.ty),
            })
        }
    }
}
