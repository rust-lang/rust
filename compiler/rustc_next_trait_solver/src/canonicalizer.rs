use std::cmp::Ordering;

use rustc_type_ir::data_structures::{HashMap, ensure_sufficient_stack};
use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::{Goal, QueryInput};
use rustc_type_ir::{
    self as ty, Canonical, CanonicalTyVarKind, CanonicalVarKind, Flags, InferCtxtLike, Interner,
    TypeFlags, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};

use crate::delegate::SolverDelegate;

/// Does this have infer/placeholder/param, free regions or ReErased?
const NEEDS_CANONICAL: TypeFlags = TypeFlags::from_bits(
    TypeFlags::HAS_INFER.bits()
        | TypeFlags::HAS_PLACEHOLDER.bits()
        | TypeFlags::HAS_PARAM.bits()
        | TypeFlags::HAS_FREE_REGIONS.bits()
        | TypeFlags::HAS_RE_ERASED.bits(),
)
.unwrap();

/// Whether we're canonicalizing a query input or the query response.
///
/// When canonicalizing an input we're in the context of the caller
/// while canonicalizing the response happens in the context of the
/// query.
#[derive(Debug, Clone, Copy)]
enum CanonicalizeMode {
    /// When canonicalizing the `param_env`, we keep `'static` as merging
    /// trait candidates relies on it when deciding whether a where-bound
    /// is trivial.
    Input { keep_static: bool },
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

pub struct Canonicalizer<'a, D: SolverDelegate<Interner = I>, I: Interner> {
    delegate: &'a D,

    // Immutable field.
    canonicalize_mode: CanonicalizeMode,

    // Mutable fields.
    variables: &'a mut Vec<I::GenericArg>,
    var_kinds: Vec<CanonicalVarKind<I>>,
    variable_lookup_table: HashMap<I::GenericArg, usize>,
    binder_index: ty::DebruijnIndex,

    /// We only use the debruijn index during lookup. We don't need to
    /// track the `variables` as each generic arg only results in a single
    /// bound variable regardless of how many times it is encountered.
    cache: HashMap<(ty::DebruijnIndex, I::Ty), I::Ty>,
}

impl<'a, D: SolverDelegate<Interner = I>, I: Interner> Canonicalizer<'a, D, I> {
    pub fn canonicalize_response<T: TypeFoldable<I>>(
        delegate: &'a D,
        max_input_universe: ty::UniverseIndex,
        variables: &'a mut Vec<I::GenericArg>,
        value: T,
    ) -> ty::Canonical<I, T> {
        let mut canonicalizer = Canonicalizer {
            delegate,
            canonicalize_mode: CanonicalizeMode::Response { max_input_universe },

            variables,
            variable_lookup_table: Default::default(),
            var_kinds: Vec::new(),
            binder_index: ty::INNERMOST,

            cache: Default::default(),
        };

        let value = if value.has_type_flags(NEEDS_CANONICAL) {
            value.fold_with(&mut canonicalizer)
        } else {
            value
        };
        assert!(!value.has_infer(), "unexpected infer in {value:?}");
        assert!(!value.has_placeholders(), "unexpected placeholders in {value:?}");
        let (max_universe, variables) = canonicalizer.finalize();
        Canonical { max_universe, variables, value }
    }

    /// When canonicalizing query inputs, we keep `'static` in the `param_env`
    /// but erase it everywhere else. We generally don't want to depend on region
    /// identity, so while it should not matter whether `'static` is kept in the
    /// value or opaque type storage as well, this prevents us from accidentally
    /// relying on it in the future.
    ///
    /// We want to keep the option of canonicalizing `'static` to an existential
    /// variable in the future by changing the way we detect global where-bounds.
    pub fn canonicalize_input<P: TypeFoldable<I>>(
        delegate: &'a D,
        variables: &'a mut Vec<I::GenericArg>,
        input: QueryInput<I, P>,
    ) -> ty::Canonical<I, QueryInput<I, P>> {
        // First canonicalize the `param_env` while keeping `'static`
        let mut env_canonicalizer = Canonicalizer {
            delegate,
            canonicalize_mode: CanonicalizeMode::Input { keep_static: true },

            variables,
            variable_lookup_table: Default::default(),
            var_kinds: Vec::new(),
            binder_index: ty::INNERMOST,

            cache: Default::default(),
        };

        let param_env = input.goal.param_env;
        let param_env = if param_env.has_type_flags(NEEDS_CANONICAL) {
            param_env.fold_with(&mut env_canonicalizer)
        } else {
            param_env
        };

        debug_assert_eq!(env_canonicalizer.binder_index, ty::INNERMOST);
        // Then canonicalize the rest of the input without keeping `'static`
        // while *mostly* reusing the canonicalizer from above.
        let mut rest_canonicalizer = Canonicalizer {
            delegate,
            canonicalize_mode: CanonicalizeMode::Input { keep_static: false },

            variables: env_canonicalizer.variables,
            // We're able to reuse the `variable_lookup_table` as whether or not
            // it already contains an entry for `'static` does not matter.
            variable_lookup_table: env_canonicalizer.variable_lookup_table,
            var_kinds: env_canonicalizer.var_kinds,
            binder_index: ty::INNERMOST,

            // We do not reuse the cache as it may contain entries whose canonicalized
            // value contains `'static`. While we could alternatively handle this by
            // checking for `'static` when using cached entries, this does not
            // feel worth the effort. I do not expect that a `ParamEnv` will ever
            // contain large enough types for caching to be necessary.
            cache: Default::default(),
        };

        let predicate = input.goal.predicate;
        let predicate = if predicate.has_type_flags(NEEDS_CANONICAL) {
            predicate.fold_with(&mut rest_canonicalizer)
        } else {
            predicate
        };
        let goal = Goal { param_env, predicate };

        let predefined_opaques_in_body = input.predefined_opaques_in_body;
        let predefined_opaques_in_body =
            if input.predefined_opaques_in_body.has_type_flags(NEEDS_CANONICAL) {
                predefined_opaques_in_body.fold_with(&mut rest_canonicalizer)
            } else {
                predefined_opaques_in_body
            };

        let value = QueryInput { goal, predefined_opaques_in_body };

        assert!(!value.has_infer(), "unexpected infer in {value:?}");
        assert!(!value.has_placeholders(), "unexpected placeholders in {value:?}");
        let (max_universe, variables) = rest_canonicalizer.finalize();
        Canonical { max_universe, variables, value }
    }

    fn get_or_insert_bound_var(
        &mut self,
        arg: impl Into<I::GenericArg>,
        kind: CanonicalVarKind<I>,
    ) -> ty::BoundVar {
        // FIXME: 16 is made up and arbitrary. We should look at some
        // perf data here.
        let arg = arg.into();
        let idx = if self.variables.len() > 16 {
            if self.variable_lookup_table.is_empty() {
                self.variable_lookup_table.extend(self.variables.iter().copied().zip(0..));
            }

            *self.variable_lookup_table.entry(arg).or_insert_with(|| {
                let var = self.variables.len();
                self.variables.push(arg);
                self.var_kinds.push(kind);
                var
            })
        } else {
            self.variables.iter().position(|&v| v == arg).unwrap_or_else(|| {
                let var = self.variables.len();
                self.variables.push(arg);
                self.var_kinds.push(kind);
                var
            })
        };

        ty::BoundVar::from(idx)
    }

    fn finalize(self) -> (ty::UniverseIndex, I::CanonicalVarKinds) {
        let mut var_kinds = self.var_kinds;
        // See the rustc-dev-guide section about how we deal with universes
        // during canonicalization in the new solver.
        match self.canonicalize_mode {
            // We try to deduplicate as many query calls as possible and hide
            // all information which should not matter for the solver.
            //
            // For this we compress universes as much as possible.
            CanonicalizeMode::Input { .. } => {}
            // When canonicalizing a response we map a universes already entered
            // by the caller to the root universe and only return useful universe
            // information for placeholders and inference variables created inside
            // of the query.
            CanonicalizeMode::Response { max_input_universe } => {
                for var in var_kinds.iter_mut() {
                    let uv = var.universe();
                    let new_uv = ty::UniverseIndex::from(
                        uv.index().saturating_sub(max_input_universe.index()),
                    );
                    *var = var.with_updated_universe(new_uv);
                }
                let max_universe = var_kinds
                    .iter()
                    .map(|kind| kind.universe())
                    .max()
                    .unwrap_or(ty::UniverseIndex::ROOT);

                let var_kinds = self.delegate.cx().mk_canonical_var_kinds(&var_kinds);
                return (max_universe, var_kinds);
            }
        }

        // Given a `var_kinds` with existentials `En` and universals `Un` in
        // universes `n`, this algorithm compresses them in place so that:
        //
        // - the new universe indices are as small as possible
        // - we create a new universe if we would otherwise
        //   1. put existentials from a different universe into the same one
        //   2. put a placeholder in the same universe as an existential which cannot name it
        //
        // Let's walk through an example:
        // - var_kinds: [E0, U1, E5, U2, E2, E6, U6], curr_compressed_uv: 0, next_orig_uv: 0
        // - var_kinds: [E0, U1, E5, U2, E2, E6, U6], curr_compressed_uv: 0, next_orig_uv: 1
        // - var_kinds: [E0, U1, E5, U2, E2, E6, U6], curr_compressed_uv: 1, next_orig_uv: 2
        // - var_kinds: [E0, U1, E5, U1, E1, E6, U6], curr_compressed_uv: 1, next_orig_uv: 5
        // - var_kinds: [E0, U1, E2, U1, E1, E6, U6], curr_compressed_uv: 2, next_orig_uv: 6
        // - var_kinds: [E0, U1, E1, U1, E1, E3, U3], curr_compressed_uv: 2, next_orig_uv: -
        //
        // This algorithm runs in `O(mn)` where `n` is the number of different universes and
        // `m` the number of variables. This should be fine as both are expected to be small.
        let mut curr_compressed_uv = ty::UniverseIndex::ROOT;
        let mut existential_in_new_uv = None;
        let mut next_orig_uv = Some(ty::UniverseIndex::ROOT);
        while let Some(orig_uv) = next_orig_uv.take() {
            let mut update_uv = |var: &mut CanonicalVarKind<I>, orig_uv, is_existential| {
                let uv = var.universe();
                match uv.cmp(&orig_uv) {
                    Ordering::Less => (), // Already updated
                    Ordering::Equal => {
                        if is_existential {
                            if existential_in_new_uv.is_some_and(|uv| uv < orig_uv) {
                                // Condition 1.
                                //
                                // We already put an existential from a outer universe
                                // into the current compressed universe, so we need to
                                // create a new one.
                                curr_compressed_uv = curr_compressed_uv.next_universe();
                            }

                            // `curr_compressed_uv` will now contain an existential from
                            // `orig_uv`. Trying to canonicalizing an existential from
                            // a higher universe has to therefore use a new compressed
                            // universe.
                            existential_in_new_uv = Some(orig_uv);
                        } else if existential_in_new_uv.is_some() {
                            // Condition 2.
                            //
                            //  `var` is a placeholder from a universe which is not nameable
                            // by an existential which we already put into the compressed
                            // universe `curr_compressed_uv`. We therefore have to create a
                            // new universe for `var`.
                            curr_compressed_uv = curr_compressed_uv.next_universe();
                            existential_in_new_uv = None;
                        }

                        *var = var.with_updated_universe(curr_compressed_uv);
                    }
                    Ordering::Greater => {
                        // We can ignore this variable in this iteration. We only look at
                        // universes which actually occur in the input for performance.
                        //
                        // For this we set `next_orig_uv` to the next smallest, not yet compressed,
                        // universe of the input.
                        if next_orig_uv.is_none_or(|curr_next_uv| uv.cannot_name(curr_next_uv)) {
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
                for var in var_kinds.iter_mut() {
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

        // We put all regions into a separate universe.
        let mut first_region = true;
        for var in var_kinds.iter_mut() {
            if var.is_region() {
                if first_region {
                    first_region = false;
                    curr_compressed_uv = curr_compressed_uv.next_universe();
                }
                assert!(var.is_existential());
                *var = var.with_updated_universe(curr_compressed_uv);
            }
        }

        let var_kinds = self.delegate.cx().mk_canonical_var_kinds(&var_kinds);
        (curr_compressed_uv, var_kinds)
    }

    fn cached_fold_ty(&mut self, t: I::Ty) -> I::Ty {
        let kind = match t.kind() {
            ty::Infer(i) => match i {
                ty::TyVar(vid) => {
                    assert_eq!(
                        self.delegate.opportunistic_resolve_ty_var(vid),
                        t,
                        "ty vid should have been resolved fully before canonicalization"
                    );

                    CanonicalVarKind::Ty(CanonicalTyVarKind::General(
                        self.delegate
                            .universe_of_ty(vid)
                            .unwrap_or_else(|| panic!("ty var should have been resolved: {t:?}")),
                    ))
                }
                ty::IntVar(vid) => {
                    assert_eq!(
                        self.delegate.opportunistic_resolve_int_var(vid),
                        t,
                        "ty vid should have been resolved fully before canonicalization"
                    );
                    CanonicalVarKind::Ty(CanonicalTyVarKind::Int)
                }
                ty::FloatVar(vid) => {
                    assert_eq!(
                        self.delegate.opportunistic_resolve_float_var(vid),
                        t,
                        "ty vid should have been resolved fully before canonicalization"
                    );
                    CanonicalVarKind::Ty(CanonicalTyVarKind::Float)
                }
                ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {
                    panic!("fresh vars not expected in canonicalization")
                }
            },
            ty::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderTy(
                    PlaceholderLike::new(placeholder.universe(), self.variables.len().into()),
                ),
                CanonicalizeMode::Response { .. } => CanonicalVarKind::PlaceholderTy(placeholder),
            },
            ty::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderTy(
                    PlaceholderLike::new(ty::UniverseIndex::ROOT, self.variables.len().into()),
                ),
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
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::Pat(_, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(..)
            | ty::UnsafeBinder(_)
            | ty::Dynamic(_, _, _)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Alias(_, _)
            | ty::Bound(_, _)
            | ty::Error(_) => {
                return if t.has_type_flags(NEEDS_CANONICAL) {
                    ensure_sufficient_stack(|| t.super_fold_with(self))
                } else {
                    t
                };
            }
        };

        let var = self.get_or_insert_bound_var(t, kind);

        Ty::new_anon_bound(self.cx(), self.binder_index, var)
    }
}

impl<D: SolverDelegate<Interner = I>, I: Interner> TypeFolder<I> for Canonicalizer<'_, D, I> {
    fn cx(&self) -> I {
        self.delegate.cx()
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T>
    where
        T: TypeFoldable<I>,
    {
        self.binder_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        let kind = match r.kind() {
            ty::ReBound(..) => return r,

            // We don't canonicalize `ReStatic` in the `param_env` as we use it
            // when checking whether a `ParamEnv` candidate is global.
            ty::ReStatic => match self.canonicalize_mode {
                CanonicalizeMode::Input { keep_static: false } => {
                    CanonicalVarKind::Region(ty::UniverseIndex::ROOT)
                }
                CanonicalizeMode::Input { keep_static: true }
                | CanonicalizeMode::Response { .. } => return r,
            },

            // `ReErased` should only be encountered in the hidden
            // type of an opaque for regions that are ignored for the purposes of
            // captures.
            //
            // FIXME: We should investigate the perf implications of not uniquifying
            // `ReErased`. We may be able to short-circuit registering region
            // obligations if we encounter a `ReErased` on one side, for example.
            ty::ReErased | ty::ReError(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => return r,
            },

            ty::ReEarlyParam(_) | ty::ReLateParam(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => {
                    panic!("unexpected region in response: {r:?}")
                }
            },

            ty::RePlaceholder(placeholder) => match self.canonicalize_mode {
                // We canonicalize placeholder regions as existentials in query inputs.
                CanonicalizeMode::Input { .. } => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
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
                    self.delegate.opportunistic_resolve_lt_var(vid),
                    r,
                    "region vid should have been resolved fully before canonicalization"
                );
                match self.canonicalize_mode {
                    CanonicalizeMode::Input { keep_static: _ } => {
                        CanonicalVarKind::Region(ty::UniverseIndex::ROOT)
                    }
                    CanonicalizeMode::Response { .. } => {
                        CanonicalVarKind::Region(self.delegate.universe_of_lt(vid).unwrap())
                    }
                }
            }
        };

        let var = self.get_or_insert_bound_var(r, kind);

        Region::new_anon_bound(self.cx(), self.binder_index, var)
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        if let Some(&ty) = self.cache.get(&(self.binder_index, t)) {
            ty
        } else {
            let res = self.cached_fold_ty(t);
            assert!(self.cache.insert((self.binder_index, t), res).is_none());
            res
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        let kind = match c.kind() {
            ty::ConstKind::Infer(i) => match i {
                ty::InferConst::Var(vid) => {
                    assert_eq!(
                        self.delegate.opportunistic_resolve_ct_var(vid),
                        c,
                        "const vid should have been resolved fully before canonicalization"
                    );
                    CanonicalVarKind::Const(self.delegate.universe_of_ct(vid).unwrap())
                }
                ty::InferConst::Fresh(_) => todo!(),
            },
            ty::ConstKind::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderConst(
                    PlaceholderLike::new(placeholder.universe(), self.variables.len().into()),
                ),
                CanonicalizeMode::Response { .. } => {
                    CanonicalVarKind::PlaceholderConst(placeholder)
                }
            },
            ty::ConstKind::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderConst(
                    PlaceholderLike::new(ty::UniverseIndex::ROOT, self.variables.len().into()),
                ),
                CanonicalizeMode::Response { .. } => panic!("param ty in response: {c:?}"),
            },
            // FIXME: See comment above -- we could fold the region separately or something.
            ty::ConstKind::Bound(_, _)
            | ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Error(_)
            | ty::ConstKind::Expr(_) => {
                return if c.has_type_flags(NEEDS_CANONICAL) { c.super_fold_with(self) } else { c };
            }
        };

        let var = self.get_or_insert_bound_var(c, kind);

        Const::new_anon_bound(self.cx(), self.binder_index, var)
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate {
        if p.flags().intersects(NEEDS_CANONICAL) { p.super_fold_with(self) } else { p }
    }
}
