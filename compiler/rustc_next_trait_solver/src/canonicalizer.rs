use std::cmp::Ordering;

use rustc_type_ir::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_type_ir::{
    self as ty, Canonical, CanonicalTyVarKind, CanonicalVarInfo, CanonicalVarKind, ConstTy,
    InferCtxtLike, Interner, IntoKind, PlaceholderLike,
};

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

pub struct Canonicalizer<'a, Infcx: InferCtxtLike<Interner = I>, I: Interner> {
    infcx: &'a Infcx,
    canonicalize_mode: CanonicalizeMode,

    variables: &'a mut Vec<I::GenericArg>,
    primitive_var_infos: Vec<CanonicalVarInfo<I>>,
    binder_index: ty::DebruijnIndex,
}

impl<'a, Infcx: InferCtxtLike<Interner = I>, I: Interner> Canonicalizer<'a, Infcx, I> {
    pub fn canonicalize<T: TypeFoldable<I>>(
        infcx: &'a Infcx,
        canonicalize_mode: CanonicalizeMode,
        variables: &'a mut Vec<I::GenericArg>,
        value: T,
    ) -> ty::Canonical<I, T> {
        let mut canonicalizer = Canonicalizer {
            infcx,
            canonicalize_mode,

            variables,
            primitive_var_infos: Vec::new(),
            binder_index: ty::INNERMOST,
        };

        let value = value.fold_with(&mut canonicalizer);
        // FIXME: Restore these assertions. Should we uplift type flags?
        // assert!(!value.has_infer(), "unexpected infer in {value:?}");
        // assert!(!value.has_placeholders(), "unexpected placeholders in {value:?}");

        let (max_universe, variables) = canonicalizer.finalize();

        Canonical { max_universe, variables, value }
    }

    fn finalize(self) -> (ty::UniverseIndex, I::CanonicalVars) {
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

                let var_infos = self.infcx.interner().mk_canonical_var_infos(&var_infos);
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
        // This algorithm runs in `O(n²)` where `n` is the number of different universe
        // indices in the input. This should be fine as `n` is expected to be small.
        let mut curr_compressed_uv = ty::UniverseIndex::ROOT;
        let mut existential_in_new_uv = false;
        let mut next_orig_uv = Some(ty::UniverseIndex::ROOT);
        while let Some(orig_uv) = next_orig_uv.take() {
            let mut update_uv = |var: &mut CanonicalVarInfo<I>, orig_uv, is_existential| {
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

        let var_infos = self.infcx.interner().mk_canonical_var_infos(&var_infos);
        (curr_compressed_uv, var_infos)
    }
}

impl<Infcx: InferCtxtLike<Interner = I>, I: Interner> TypeFolder<I>
    for Canonicalizer<'_, Infcx, I>
{
    fn interner(&self) -> I {
        self.infcx.interner()
    }

    fn fold_binder<T>(&mut self, t: I::Binder<T>) -> I::Binder<T>
    where
        T: TypeFoldable<I>,
        I::Binder<T>: TypeSuperFoldable<I>,
    {
        self.binder_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        let kind = match r.kind() {
            ty::ReBound(..) => return r,

            // We may encounter `ReStatic` in item signatures or the hidden type
            // of an opaque. `ReErased` should only be encountered in the hidden
            // type of an opaque for regions that are ignored for the purposes of
            // captures.
            //
            // FIXME: We should investigate the perf implications of not uniquifying
            // `ReErased`. We may be able to short-circuit registering region
            // obligations if we encounter a `ReErased` on one side, for example.
            ty::ReStatic | ty::ReErased => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => return r,
            },

            ty::ReEarlyParam(_) | ty::ReLateParam(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => {
                    panic!("unexpected region in response: {r:?}")
                }
            },

            ty::RePlaceholder(placeholder) => match self.canonicalize_mode {
                // We canonicalize placeholder regions as existentials in query inputs.
                CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { max_input_universe } => {
                    // If we have a placeholder region inside of a query, it must be from
                    // a new universe.
                    if max_input_universe.can_name(placeholder.universe()) {
                        panic!("new placeholder in universe {max_input_universe:?}: {r:?}");
                    }
                    CanonicalVarKind::PlaceholderRegion(placeholder)
                }
            },

            ty::ReVar(vid) => {
                assert_eq!(
                    self.infcx.opportunistic_resolve_lt_var(vid),
                    None,
                    "region vid should have been resolved fully before canonicalization"
                );
                match self.canonicalize_mode {
                    CanonicalizeMode::Input => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                    CanonicalizeMode::Response { .. } => {
                        CanonicalVarKind::Region(self.infcx.universe_of_lt(vid).unwrap())
                    }
                }
            }
            ty::ReError(_) => return r,
        };

        let existing_bound_var = match self.canonicalize_mode {
            CanonicalizeMode::Input => None,
            CanonicalizeMode::Response { .. } => {
                self.variables.iter().position(|&v| v == r.into()).map(ty::BoundVar::from)
            }
        };

        let var = existing_bound_var.unwrap_or_else(|| {
            let var = ty::BoundVar::from(self.variables.len());
            self.variables.push(r.into());
            self.primitive_var_infos.push(CanonicalVarInfo { kind });
            var
        });

        self.interner().mk_bound_region(self.binder_index, var)
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty
    where
        I::Ty: TypeSuperFoldable<I>,
    {
        let kind = match t.kind() {
            ty::Infer(i) => match i {
                ty::TyVar(vid) => {
                    assert_eq!(
                        self.infcx.root_ty_var(vid),
                        vid,
                        "ty vid should have been resolved fully before canonicalization"
                    );
                    assert_eq!(
                        self.infcx.probe_ty_var(vid),
                        None,
                        "ty vid should have been resolved fully before canonicalization"
                    );

                    CanonicalVarKind::Ty(CanonicalTyVarKind::General(
                        self.infcx
                            .universe_of_ty(vid)
                            .unwrap_or_else(|| panic!("ty var should have been resolved: {t:?}")),
                    ))
                }
                ty::IntVar(_) => CanonicalVarKind::Ty(CanonicalTyVarKind::Int),
                ty::FloatVar(_) => CanonicalVarKind::Ty(CanonicalTyVarKind::Float),
                ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {
                    todo!()
                }
            },
            ty::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderTy(PlaceholderLike::new(
                    placeholder.universe(),
                    self.variables.len().into(),
                )),
                CanonicalizeMode::Response { .. } => CanonicalVarKind::PlaceholderTy(placeholder),
            },
            ty::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderTy(PlaceholderLike::new(
                    ty::UniverseIndex::ROOT,
                    self.variables.len().into(),
                )),
                CanonicalizeMode::Response { .. } => panic!("param ty in response: {t:?}"),
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
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
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

        self.interner().mk_bound_ty(self.binder_index, var)
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const
    where
        I::Const: TypeSuperFoldable<I>,
    {
        let kind = match c.kind() {
            ty::ConstKind::Infer(i) => {
                // FIXME: we should fold the ty too eventually
                match i {
                    ty::InferConst::Var(vid) => {
                        assert_eq!(
                            self.infcx.root_ct_var(vid),
                            vid,
                            "region vid should have been resolved fully before canonicalization"
                        );
                        assert_eq!(
                            self.infcx.probe_ct_var(vid),
                            None,
                            "region vid should have been resolved fully before canonicalization"
                        );
                        CanonicalVarKind::Const(self.infcx.universe_of_ct(vid).unwrap(), c.ty())
                    }
                    ty::InferConst::EffectVar(_) => CanonicalVarKind::Effect,
                    ty::InferConst::Fresh(_) => todo!(),
                }
            }
            ty::ConstKind::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderConst(
                    PlaceholderLike::new(placeholder.universe(), self.variables.len().into()),
                    c.ty(),
                ),
                CanonicalizeMode::Response { .. } => {
                    CanonicalVarKind::PlaceholderConst(placeholder, c.ty())
                }
            },
            ty::ConstKind::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input => CanonicalVarKind::PlaceholderConst(
                    PlaceholderLike::new(ty::UniverseIndex::ROOT, self.variables.len().into()),
                    c.ty(),
                ),
                CanonicalizeMode::Response { .. } => panic!("param ty in response: {c:?}"),
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

        self.interner().mk_bound_const(self.binder_index, var, c.ty())
    }
}
