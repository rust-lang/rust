use std::cmp::Ordering;

use crate::infer::InferCtxt;
use rustc_middle::infer::canonical::Canonical;
use rustc_middle::infer::canonical::CanonicalTyVarKind;
use rustc_middle::infer::canonical::CanonicalVarInfo;
use rustc_middle::infer::canonical::CanonicalVarInfos;
use rustc_middle::infer::canonical::CanonicalVarKind;
use rustc_middle::ty::BoundRegionKind::BrAnon;
use rustc_middle::ty::BoundTyKind;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::TypeVisitableExt;
use rustc_middle::ty::{self, Ty};
use rustc_middle::ty::{TypeFoldable, TypeFolder, TypeSuperFoldable};

/// Whether we're canonicalizing a query input or the query response.
///
/// When canonicalizing an input we're in the context of the caller
/// while canonicalizing the response happens in the context of the
/// query.
#[derive(Debug, Clone, Copy)]
pub enum CanonicalizeMode {
    Input,
    /// FIXME: We currently return region constraints referring to
    /// placeholders and inference variables from a binder instantiated
    /// inside of the query.
    ///
    /// In the long term we should eagerly deal with these constraints
    /// inside of the query and only propagate constraints which are
    /// actually nameable by the caller.
    Response {
        /// The highest universe nameable by the caller.
        ///
        /// All variables in a universe nameable by the caller get mapped
        /// to the root universe in the response and then mapped back to
        /// their correct universe when applying the query response in the
        /// context of the caller.
        ///
        /// This doesn't work for universes created inside of the query so
        /// we do remember their universe in the response.
        max_input_universe: ty::UniverseIndex,
    },
}

pub struct Canonicalizer<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    canonicalize_mode: CanonicalizeMode,

    variables: &'a mut Vec<ty::GenericArg<'tcx>>,
    primitive_var_infos: Vec<CanonicalVarInfo<'tcx>>,
    binder_index: ty::DebruijnIndex,
}

impl<'a, 'tcx> Canonicalizer<'a, 'tcx> {
    #[instrument(level = "debug", skip(infcx), ret)]
    pub fn canonicalize<T: TypeFoldable<TyCtxt<'tcx>>>(
        infcx: &'a InferCtxt<'tcx>,
        canonicalize_mode: CanonicalizeMode,
        variables: &'a mut Vec<ty::GenericArg<'tcx>>,
        value: T,
    ) -> Canonical<'tcx, T> {
        let mut canonicalizer = Canonicalizer {
            infcx,
            canonicalize_mode,

            variables,
            primitive_var_infos: Vec::new(),
            binder_index: ty::INNERMOST,
        };

        let value = value.fold_with(&mut canonicalizer);
        assert!(!value.has_infer());
        assert!(!value.has_placeholders());

        let (max_universe, variables) = canonicalizer.finalize();

        Canonical { max_universe, variables, value }
    }

    fn finalize(self) -> (ty::UniverseIndex, CanonicalVarInfos<'tcx>) {
        let mut var_infos = self.primitive_var_infos;
        // See the rustc-dev-guide section about how we deal with universes
        // during canonicalization in the new solver.
        match self.canonicalize_mode {
            // We try to deduplicate as many query calls as possible and hide
            // all information which should not matter for the solver.
            //
            // For this we compress universes as much as possible.
            CanonicalizeMode::Input => {}
            // When canonicalizing a response we map a universes already entered
            // by the caller to the root universe and only return useful universe
            // information for placeholders and inference variables created inside
            // of the query.
            CanonicalizeMode::Response { max_input_universe } => {
                for var in var_infos.iter_mut() {
                    let uv = var.universe();
                    let new_uv = ty::UniverseIndex::from(
                        uv.index().saturating_sub(max_input_universe.index()),
                    );
                    *var = var.with_updated_universe(new_uv);
                }
                let max_universe = var_infos
                    .iter()
                    .map(|info| info.universe())
                    .max()
                    .unwrap_or(ty::UniverseIndex::ROOT);

                let var_infos = self.infcx.tcx.mk_canonical_var_infos(&var_infos);
                return (max_universe, var_infos);
            }
        }

        // Given a `var_infos` with existentials `En` and universals `Un` in
        // universes `n`, this algorithm compresses them in place so that:
        //
        // - the new universe indices are as small as possible
        // - we only create a new universe if we would otherwise put a placeholder in
        //   the same compressed universe as an existential which cannot name it
        //
        // Let's walk through an example:
        // - var_infos: [E0, U1, E5, U2, E2, E6, U6], curr_compressed_uv: 0, next_orig_uv: 0
        // - var_infos: [E0, U1, E5, U2, E2, E6, U6], curr_compressed_uv: 0, next_orig_uv: 1
        // - var_infos: [E0, U1, E5, U2, E2, E6, U6], curr_compressed_uv: 1, next_orig_uv: 2
        // - var_infos: [E0, U1, E5, U1, E1, E6, U6], curr_compressed_uv: 1, next_orig_uv: 5
        // - var_infos: [E0, U1, E1, U1, E1, E6, U6], curr_compressed_uv: 1, next_orig_uv: 6
        // - var_infos: [E0, U1, E1, U1, E1, E2, U2], curr_compressed_uv: 2, next_orig_uv: -
        //
        // This algorithm runs in `O(nm)` where `n` is the number of different universe
        // indices in the input and `m` is the number of canonical variables.
        // This should be fine as both `n` and `m` are expected to be small.
        let mut curr_compressed_uv = ty::UniverseIndex::ROOT;
        let mut existential_in_new_uv = false;
        let mut next_orig_uv = Some(ty::UniverseIndex::ROOT);
        while let Some(orig_uv) = next_orig_uv.take() {
            let mut update_uv = |var: &mut CanonicalVarInfo<'tcx>, orig_uv, is_existential| {
                let uv = var.universe();
                match uv.cmp(&orig_uv) {
                    Ordering::Less => (), // Already updated
                    Ordering::Equal => {
                        if is_existential {
                            existential_in_new_uv = true;
                        } else if existential_in_new_uv {
                            //  `var` is a placeholder from a universe which is not nameable
                            // by an existential which we already put into the compressed
                            // universe `curr_compressed_uv`. We therefore have to create a
                            // new universe for `var`.
                            curr_compressed_uv = curr_compressed_uv.next_universe();
                            existential_in_new_uv = false;
                        }

                        *var = var.with_updated_universe(curr_compressed_uv);
                    }
                    Ordering::Greater => {
                        // We can ignore this variable in this iteration. We only look at
                        // universes which actually occur in the input for performance.
                        //
                        // For this we set `next_orig_uv` to the next smallest, not yet compressed,
                        // universe of the input.
                        if next_orig_uv.map_or(true, |curr_next_uv| uv.cannot_name(curr_next_uv)) {
                            next_orig_uv = Some(uv);
                        }
                    }
                }
            };

            // For each universe which occurs in the input, we first iterate over all
            // placeholders and then over all inference variables.
            //
            // Whenever we compress the universe of a placeholder, no existential with
            // an already compressed universe can name that placeholder.
            for is_existential in [false, true] {
                for var in var_infos.iter_mut() {
                    // We simply put all regions from the input into the highest
                    // compressed universe, so we only deal with them at the end.
                    if !var.is_region() {
                        if is_existential == var.is_existential() {
                            update_uv(var, orig_uv, is_existential)
                        }
                    }
                }
            }
        }

        for var in var_infos.iter_mut() {
            if var.is_region() {
                assert!(var.is_existential());
                *var = var.with_updated_universe(curr_compressed_uv);
            }
        }

        let var_infos = self.infcx.tcx.mk_canonical_var_infos(&var_infos);
        (curr_compressed_uv, var_infos)
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for Canonicalizer<'_, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
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

    fn fold_region(&mut self, mut r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match self.canonicalize_mode {
            CanonicalizeMode::Input => {
                // Don't resolve infer vars in input, since it affects
                // caching and may cause trait selection bugs which rely
                // on regions to be equal.
            }
            CanonicalizeMode::Response { .. } => {
                if let ty::ReVar(vid) = *r {
                    r = self
                        .infcx
                        .inner
                        .borrow_mut()
                        .unwrap_region_constraints()
                        .opportunistic_resolve_var(self.infcx.tcx, vid);
                }
            }
        }

        let kind = match *r {
            ty::ReLateBound(..) => return r,

            ty::ReStatic => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => return r,
            },

            ty::ReErased | ty::ReFree(_) | ty::ReEarlyBound(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => bug!("unexpected region in response: {r:?}"),
            },

            ty::RePlaceholder(placeholder) => match self.canonicalize_mode {
                // We canonicalize placeholder regions as existentials in query inputs.
                CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { max_input_universe } => {
                    // If we have a placeholder region inside of a query, it must be from
                    // a new universe.
                    if max_input_universe.can_name(placeholder.universe) {
                        bug!("new placeholder in universe {max_input_universe:?}: {r:?}");
                    }
                    CanonicalVarKind::PlaceholderRegion(placeholder)
                }
            },

            ty::ReVar(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => {
                    CanonicalVarKind::Region(self.infcx.universe_of_region(r))
                }
            },

            ty::ReError(_) => return r,
        };

        let var = ty::BoundVar::from(
            self.variables.iter().position(|&v| v == r.into()).unwrap_or_else(|| {
                let var = self.variables.len();
                self.variables.push(r.into());
                self.primitive_var_infos.push(CanonicalVarInfo { kind });
                var
            }),
        );
        let br = ty::BoundRegion { var, kind: BrAnon(None) };
        ty::Region::new_late_bound(self.interner(), self.binder_index, br)
    }

    fn fold_ty(&mut self, mut t: Ty<'tcx>) -> Ty<'tcx> {
        let kind = match *t.kind() {
            ty::Infer(ty::TyVar(mut vid)) => {
                // We need to canonicalize the *root* of our ty var.
                // This is so that our canonical response correctly reflects
                // any equated inference vars correctly!
                let root_vid = self.infcx.root_var(vid);
                if root_vid != vid {
                    t = self.infcx.tcx.mk_ty_var(root_vid);
                    vid = root_vid;
                }

                match self.infcx.probe_ty_var(vid) {
                    Ok(t) => return self.fold_ty(t),
                    Err(ui) => CanonicalVarKind::Ty(CanonicalTyVarKind::General(ui)),
                }
            }
            ty::Infer(ty::IntVar(vid)) => {
                let nt = self.infcx.opportunistic_resolve_int_var(vid);
                if nt != t {
                    return self.fold_ty(nt);
                } else {
                    CanonicalVarKind::Ty(CanonicalTyVarKind::Int)
                }
            }
            ty::Infer(ty::FloatVar(vid)) => {
                let nt = self.infcx.opportunistic_resolve_float_var(vid);
                if nt != t {
                    return self.fold_ty(nt);
                } else {
                    CanonicalVarKind::Ty(CanonicalTyVarKind::Float)
                }
            }
            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("fresh var during canonicalization: {t:?}")
            }
            ty::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderTy(ty::Placeholder {
                    universe: placeholder.universe,
                    bound: ty::BoundTy {
                        var: ty::BoundVar::from_usize(self.variables.len()),
                        kind: ty::BoundTyKind::Anon,
                    },
                }),
                CanonicalizeMode::Response { .. } => CanonicalVarKind::PlaceholderTy(placeholder),
            },
            ty::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderTy(ty::Placeholder {
                    universe: ty::UniverseIndex::ROOT,
                    bound: ty::BoundTy {
                        var: ty::BoundVar::from_usize(self.variables.len()),
                        kind: ty::BoundTyKind::Anon,
                    },
                }),
                CanonicalizeMode::Response { .. } => bug!("param ty in response: {t:?}"),
            },
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_)
            | ty::Dynamic(_, _, _)
            | ty::Closure(_, _)
            | ty::Generator(_, _, _)
            | ty::GeneratorWitness(_)
            | ty::GeneratorWitnessMIR(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Alias(_, _)
            | ty::Bound(_, _)
            | ty::Error(_) => return t.super_fold_with(self),
        };

        let var = ty::BoundVar::from(
            self.variables.iter().position(|&v| v == t.into()).unwrap_or_else(|| {
                let var = self.variables.len();
                self.variables.push(t.into());
                self.primitive_var_infos.push(CanonicalVarInfo { kind });
                var
            }),
        );
        let bt = ty::BoundTy { var, kind: BoundTyKind::Anon };
        self.interner().mk_bound(self.binder_index, bt)
    }

    fn fold_const(&mut self, mut c: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let kind = match c.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(mut vid)) => {
                // We need to canonicalize the *root* of our const var.
                // This is so that our canonical response correctly reflects
                // any equated inference vars correctly!
                let root_vid = self.infcx.root_const_var(vid);
                if root_vid != vid {
                    c = self.infcx.tcx.mk_const(ty::InferConst::Var(root_vid), c.ty());
                    vid = root_vid;
                }

                match self.infcx.probe_const_var(vid) {
                    Ok(c) => return self.fold_const(c),
                    Err(universe) => CanonicalVarKind::Const(universe, c.ty()),
                }
            }
            ty::ConstKind::Infer(ty::InferConst::Fresh(_)) => {
                bug!("fresh var during canonicalization: {c:?}")
            }
            ty::ConstKind::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderConst(
                    ty::Placeholder {
                        universe: placeholder.universe,
                        bound: ty::BoundVar::from(self.variables.len()),
                    },
                    c.ty(),
                ),
                CanonicalizeMode::Response { .. } => {
                    CanonicalVarKind::PlaceholderConst(placeholder, c.ty())
                }
            },
            ty::ConstKind::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderConst(
                    ty::Placeholder {
                        universe: ty::UniverseIndex::ROOT,
                        bound: ty::BoundVar::from(self.variables.len()),
                    },
                    c.ty(),
                ),
                CanonicalizeMode::Response { .. } => bug!("param ty in response: {c:?}"),
            },
            ty::ConstKind::Bound(_, _)
            | ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Error(_)
            | ty::ConstKind::Expr(_) => return c.super_fold_with(self),
        };

        let var = ty::BoundVar::from(
            self.variables.iter().position(|&v| v == c.into()).unwrap_or_else(|| {
                let var = self.variables.len();
                self.variables.push(c.into());
                self.primitive_var_infos.push(CanonicalVarInfo { kind });
                var
            }),
        );
        self.interner().mk_const(ty::ConstKind::Bound(self.binder_index, var), c.ty())
    }
}
