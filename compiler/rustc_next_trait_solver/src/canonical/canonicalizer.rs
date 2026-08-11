use std::collections::hash_map::Entry;
use std::mem;

use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::{Goal, QueryInput};
use rustc_type_ir::{
    self as ty, Canonical, CanonicalParamEnvCacheEntry, CanonicalVarKind, CanonicalizerState,
    Flags, InferCtxtLike, Interner, PlaceholderConst, PlaceholderType, Region, TypeFlags,
    TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};
use thin_vec::ThinVec;

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

#[derive(Debug, Clone, Copy)]
enum CanonicalizeInputKind {
    /// When canonicalizing the `param_env`, we keep `'static` as merging
    /// trait candidates relies on it when deciding whether a where-bound
    /// is trivial.
    ParamEnv,
    /// When canonicalizing predicates, we don't keep `'static`.
    Predicate,
}

/// Whether we're canonicalizing a query input or the query response.
///
/// When canonicalizing an input we're in the context of the caller
/// while canonicalizing the response happens in the context of the
/// query.
#[derive(Debug, Clone, Copy)]
enum CanonicalizeMode {
    Input(CanonicalizeInputKind),
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

pub(super) struct Canonicalizer<'a, D: SolverDelegate<Interner = I>, I: Interner> {
    delegate: &'a D,

    // Immutable field.
    canonicalize_mode: CanonicalizeMode,

    // Mutable fields.
    state: CanonicalizerState<I>,
}

impl<'a, D: SolverDelegate<Interner = I>, I: Interner> Canonicalizer<'a, D, I> {
    fn new(delegate: &'a D, canonicalize_mode: CanonicalizeMode) -> Self {
        Canonicalizer { delegate, canonicalize_mode, state: delegate.obtain_canonicalizer_state() }
    }

    pub(super) fn canonicalize_response<T: TypeFoldable<I>>(
        delegate: &'a D,
        max_input_universe: ty::UniverseIndex,
        value: T,
    ) -> ty::Canonical<I, T> {
        let mut canonicalizer =
            Canonicalizer::new(delegate, CanonicalizeMode::Response { max_input_universe });
        let value = if value.has_type_flags(NEEDS_CANONICAL) {
            value.fold_with(&mut canonicalizer)
        } else {
            value
        };
        debug_assert!(!value.has_infer(), "unexpected infer in {value:?}");
        debug_assert!(!value.has_placeholders(), "unexpected placeholders in {value:?}");
        let (max_universe, _variables, var_kinds) = canonicalizer.finalize();

        Canonical { max_universe, var_kinds, value }
    }

    // The return value is the canonicalized `param_env`, plus a canonicalizer suitable for
    // canonicalizing the rest of the input. (For efficiency, and when appropriate, the returned
    // canonicalizer will be the same one used on `param_env`, with suitable modifications.)
    fn canonicalize_param_env(delegate: &'a D, param_env: I::ParamEnv) -> (I::ParamEnv, Self) {
        if !param_env.has_type_flags(NEEDS_CANONICAL) {
            let rest_canonicalizer = Canonicalizer::new(
                delegate,
                CanonicalizeMode::Input(CanonicalizeInputKind::Predicate),
            );

            return (param_env, rest_canonicalizer);
        }

        // Do the `env` canonicalization, and then convert the canonicalizer to `rest` form for
        // subsequent use.
        let do_env_and_make_rest = || {
            let mut env_canonicalizer = Canonicalizer::new(
                delegate,
                CanonicalizeMode::Input(CanonicalizeInputKind::ParamEnv),
            );
            let param_env = param_env.fold_with(&mut env_canonicalizer);

            debug_assert!(env_canonicalizer.state.sub_root_lookup_table.is_empty());

            // Transform the `env_canonicalizer` into the `rest_canonicalizer`, keeping some things
            // and replacing others.
            //
            // We do not reuse the cache as it may contain entries whose canonicalized
            // value contains `'static`. While we could alternatively handle this by
            // checking for `'static` when using cached entries, this does not
            // feel worth the effort. I do not expect that a `ParamEnv` will ever
            // contain large enough types for caching to be necessary.
            //
            // We clear the cache rather than deleting it or replacing it with an empty cache. This
            // lets the allocated capacity be reused later.
            let mut rest_canonicalizer = env_canonicalizer;
            rest_canonicalizer.canonicalize_mode =
                CanonicalizeMode::Input(CanonicalizeInputKind::Predicate);
            rest_canonicalizer.state.cache.clear();

            (param_env, rest_canonicalizer)
        };

        // Check whether we can use the global cache for this param_env. As we only use
        // the `param_env` itself as the cache key, considering any additional information
        // during its canonicalization would be incorrect. We always canonicalize region
        // inference variables in a separate universe, so these are fine. However, we do
        // track the universe of type and const inference variables so these must not be
        // globally cached. We don't rely on any additional information when canonicalizing
        // placeholders.
        if !param_env.has_non_region_infer() {
            delegate.cx().with_canonical_param_env_cache(|cache| match cache.0.entry(param_env) {
                Entry::Vacant(e) => {
                    // Cache miss. Do `env` canonicalization and get `rest_canonicalizer`, and
                    // fill in the cache entry.
                    let (param_env, rest_canonicalizer) = do_env_and_make_rest();
                    e.insert(CanonicalParamEnvCacheEntry {
                        param_env,
                        variables: rest_canonicalizer.state.variables.clone(),
                        var_kinds: rest_canonicalizer.state.var_kinds.clone(),
                        // SAFETY: The iterated elements go straight back into a hashmap.
                        #[allow(rustc::potential_query_instability)]
                        variable_lookup_table: rest_canonicalizer
                            .state
                            .variable_lookup_table
                            .iter()
                            .map(|(&arg, &idx)| (arg, idx))
                            .collect(),
                    });
                    (param_env, rest_canonicalizer)
                }
                Entry::Occupied(e) => {
                    // Cache hit; no canonicalization required. Just set up `rest_canonicalizer`.
                    let e = e.get();
                    let mut rest_canonicalizer = Canonicalizer::new(
                        delegate,
                        CanonicalizeMode::Input(CanonicalizeInputKind::Predicate),
                    );
                    rest_canonicalizer.state.variables.extend(e.variables.iter().copied());
                    rest_canonicalizer.state.var_kinds.extend(e.var_kinds.iter().copied());
                    // SAFETY: The iterated elements go straight back into a hashmap.
                    #[allow(rustc::potential_query_instability)]
                    rest_canonicalizer
                        .state
                        .variable_lookup_table
                        .extend(e.variable_lookup_table.iter().map(|(&arg, &idx)| (arg, idx)));
                    (e.param_env, rest_canonicalizer)
                }
            })
        } else {
            // Do `env` canonicalization and get `rest_canonicalizer`.
            do_env_and_make_rest()
        }
    }

    /// When canonicalizing query inputs, we keep `'static` in the `param_env`
    /// but erase it everywhere else. We generally don't want to depend on region
    /// identity, so while it should not matter whether `'static` is kept in the
    /// value or opaque type storage as well, this prevents us from accidentally
    /// relying on it in the future.
    ///
    /// We want to keep the option of canonicalizing `'static` to an existential
    /// variable in the future by changing the way we detect global where-bounds.
    pub(super) fn canonicalize_input<P: TypeFoldable<I>>(
        delegate: &'a D,
        input: QueryInput<I, P>,
    ) -> (ThinVec<I::GenericArg>, ty::Canonical<I, QueryInput<I, P>>) {
        // First canonicalize the `param_env` while keeping `'static`. This produces a
        // canonicalizer that can canonicalize the rest of the input without keeping `'static`.
        let (param_env, mut rest_canonicalizer) =
            Self::canonicalize_param_env(delegate, input.goal.param_env);

        let predicate = input.goal.predicate;
        let predicate = predicate.fold_with(&mut rest_canonicalizer);
        let goal = Goal { param_env, predicate };

        let predefined_opaques_in_body = input.predefined_opaques_in_body;
        let predefined_opaques_in_body =
            if predefined_opaques_in_body.has_type_flags(NEEDS_CANONICAL) {
                predefined_opaques_in_body.fold_with(&mut rest_canonicalizer)
            } else {
                predefined_opaques_in_body
            };

        let value = QueryInput { goal, predefined_opaques_in_body };

        debug_assert!(!value.has_infer(), "unexpected infer in {value:?}");
        debug_assert!(!value.has_placeholders(), "unexpected placeholders in {value:?}");
        let (max_universe, variables, var_kinds) = rest_canonicalizer.finalize();
        (variables, Canonical { max_universe, var_kinds, value })
    }

    fn get_or_insert_bound_var(
        &mut self,
        arg: impl Into<I::GenericArg>,
        kind: CanonicalVarKind<I>,
    ) -> ty::BoundVar {
        // The exact value of 16 here doesn't matter that much (8 and 32 give extremely similar
        // results). So long as we have protection against the rare cases where the length reaches
        // 1000+ (e.g. `wg-grammar`).
        let arg = arg.into();
        let idx = if self.state.variables.len() > 16 {
            if self.state.variable_lookup_table.is_empty() {
                self.state
                    .variable_lookup_table
                    .extend(self.state.variables.iter().copied().zip(0..));
            }

            *self.state.variable_lookup_table.entry(arg).or_insert_with(|| {
                let var = self.state.variables.len();
                self.state.variables.push(arg);
                self.state.var_kinds.push(kind);
                var
            })
        } else {
            self.state.variables.iter().position(|&v| v == arg).unwrap_or_else(|| {
                let var = self.state.variables.len();
                self.state.variables.push(arg);
                self.state.var_kinds.push(kind);
                var
            })
        };

        ty::BoundVar::from(idx)
    }

    fn get_or_insert_sub_root(&mut self, vid: ty::TyVid) -> ty::BoundVar {
        let root_vid = self.delegate.sub_unification_table_root_var(vid);
        let idx = *self
            .state
            .sub_root_lookup_table
            .entry(root_vid)
            .or_insert_with(|| self.state.variables.len());
        ty::BoundVar::from(idx)
    }

    fn finalize(mut self) -> (ty::UniverseIndex, ThinVec<I::GenericArg>, I::CanonicalVarKinds) {
        // See the rustc-dev-guide section about how we deal with universes
        // during canonicalization in the new solver.
        let max_universe = match self.canonicalize_mode {
            // All placeholders and vars are canonicalized in the root universe.
            CanonicalizeMode::Input { .. } => {
                debug_assert!(
                    self.state
                        .var_kinds
                        .iter()
                        .all(|var| var.universe() == ty::UniverseIndex::ROOT),
                    "expected all vars to be canonicalized in root universe: {:#?}",
                    self.state.var_kinds,
                );
                ty::UniverseIndex::ROOT
            }
            // When canonicalizing a response we map a universes already entered
            // by the caller to the root universe and only return useful universe
            // information for placeholders and inference variables created inside
            // of the query.
            CanonicalizeMode::Response { max_input_universe } => {
                for var in self.state.var_kinds.iter_mut() {
                    let uv = var.universe();
                    let new_uv = ty::UniverseIndex::from(
                        uv.index().saturating_sub(max_input_universe.index()),
                    );
                    *var = var.with_updated_universe(new_uv);
                }
                self.state
                    .var_kinds
                    .iter()
                    .map(|kind| kind.universe())
                    .max()
                    .unwrap_or(ty::UniverseIndex::ROOT)
            }
        };
        let variables = mem::take(&mut self.state.variables);
        let var_kinds = self.delegate.cx().mk_canonical_var_kinds(&self.state.var_kinds);

        // We have finished with this canonicalizer and can return its state to the delegate for
        // later reuse.
        self.delegate.release_canonicalizer_state(self.state);

        (max_universe, variables, var_kinds)
    }

    fn inner_fold_ty(&mut self, t: I::Ty) -> I::Ty {
        let kind = match t.kind() {
            ty::Infer(i) => match i {
                ty::TyVar(vid) => {
                    debug_assert_eq!(
                        self.delegate.shallow_resolve_ty_var(vid),
                        t,
                        "ty vid should have been resolved fully before canonicalization"
                    );

                    let sub_root = self.get_or_insert_sub_root(vid);
                    let ui = match self.canonicalize_mode {
                        CanonicalizeMode::Input { .. } => ty::UniverseIndex::ROOT,
                        CanonicalizeMode::Response { .. } => self
                            .delegate
                            .universe_of_ty(vid)
                            .unwrap_or_else(|| panic!("ty var should have been resolved: {t:?}")),
                    };
                    CanonicalVarKind::Ty { ui, sub_root }
                }
                ty::IntVar(vid) => {
                    debug_assert_eq!(
                        self.delegate.shallow_resolve_int_var(vid),
                        t,
                        "ty vid should have been resolved fully before canonicalization"
                    );
                    CanonicalVarKind::Int
                }
                ty::FloatVar(vid) => {
                    debug_assert_eq!(
                        self.delegate.shallow_resolve_float_var(vid),
                        t,
                        "ty vid should have been resolved fully before canonicalization"
                    );
                    CanonicalVarKind::Float
                }
                ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {
                    panic!("fresh vars not expected in canonicalization")
                }
            },
            ty::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => {
                    CanonicalVarKind::PlaceholderTy(PlaceholderType::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
                CanonicalizeMode::Response { .. } => CanonicalVarKind::PlaceholderTy(placeholder),
            },
            ty::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => {
                    CanonicalVarKind::PlaceholderTy(PlaceholderType::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
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
            | ty::Dynamic(_, _)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Alias(_, _)
            | ty::Bound(_, _)
            | ty::Error(_) => {
                return t.super_fold_with(self);
            }
        };

        let var = self.get_or_insert_bound_var(t, kind);

        Ty::new_canonical_bound(self.cx(), var)
    }
}

impl<D: SolverDelegate<Interner = I>, I: Interner> TypeFolder<I> for Canonicalizer<'_, D, I> {
    fn cx(&self) -> I {
        self.delegate.cx()
    }

    fn fold_region(&mut self, r: Region<I>) -> Region<I> {
        // We canonicalize free regions from the input into placeholder regions so that
        // region constraints created in nested contexts can be propagated back to the
        // caller, instead of unifying them.
        // See the following Zulip discussion for details:
        // https://rust-lang.zulipchat.com/#narrow/channel/364551-t-types.2Ftrait-system-refactor/topic/A.20question.20on.20.23251/near/579240238
        let kind = match r.kind() {
            ty::ReBound(..) => return r,

            // We don't canonicalize `ReStatic` in the `param_env` as we use it
            // when checking whether a `ParamEnv` candidate is global.
            ty::ReStatic => match self.canonicalize_mode {
                CanonicalizeMode::Input(CanonicalizeInputKind::Predicate) => {
                    CanonicalVarKind::PlaceholderRegion(ty::PlaceholderRegion::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
                CanonicalizeMode::Input(CanonicalizeInputKind::ParamEnv)
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
                CanonicalizeMode::Input(_) => {
                    CanonicalVarKind::PlaceholderRegion(ty::PlaceholderRegion::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
                CanonicalizeMode::Response { .. } => return r,
            },

            ty::ReEarlyParam(_) | ty::ReLateParam(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input(_) => {
                    CanonicalVarKind::PlaceholderRegion(ty::PlaceholderRegion::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
                CanonicalizeMode::Response { .. } => {
                    panic!("unexpected region in response: {r:?}")
                }
            },

            ty::RePlaceholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input(_) => {
                    CanonicalVarKind::PlaceholderRegion(ty::PlaceholderRegion::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
                CanonicalizeMode::Response { max_input_universe } => {
                    // If we have a placeholder region inside of a query, it must be from
                    // a new universe, unless from the root universe, which is used for
                    // canonicalization of any free region from the input.
                    if placeholder.universe() != ty::UniverseIndex::ROOT
                        && max_input_universe.can_name(placeholder.universe())
                    {
                        panic!("new placeholder in universe {max_input_universe:?}: {r:?}");
                    }
                    CanonicalVarKind::PlaceholderRegion(placeholder)
                }
            },

            ty::ReVar(vid) => {
                debug_assert_eq!(
                    self.delegate.shallow_resolve_region_var(vid),
                    r,
                    "region vid should have been resolved fully before canonicalization"
                );
                match self.canonicalize_mode {
                    CanonicalizeMode::Input(_) => {
                        CanonicalVarKind::PlaceholderRegion(ty::PlaceholderRegion::new_anon(
                            ty::UniverseIndex::ROOT,
                            self.state.variables.len().into(),
                        ))
                    }
                    CanonicalizeMode::Response { .. } => {
                        CanonicalVarKind::Region(self.delegate.universe_of_region(vid).unwrap())
                    }
                }
            }
        };

        let var = self.get_or_insert_bound_var(r, kind);

        Region::new_canonical_bound(self.cx(), var)
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        if !t.flags().intersects(NEEDS_CANONICAL) {
            t
        } else if let Some(&ty) = self.state.cache.get(&t) {
            ty
        } else {
            let res = self.inner_fold_ty(t);
            let is_unseen = self.state.cache.insert(t, res);
            assert!(is_unseen);
            res
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        if !c.flags().intersects(NEEDS_CANONICAL) {
            return c;
        }

        let kind = match c.kind() {
            ty::ConstKind::Infer(i) => match i {
                ty::InferConst::Var(vid) => {
                    debug_assert_eq!(
                        self.delegate.shallow_resolve_const_var(vid),
                        c,
                        "const vid should have been resolved fully before canonicalization"
                    );

                    match self.canonicalize_mode {
                        CanonicalizeMode::Input { .. } => {
                            CanonicalVarKind::Const(ty::UniverseIndex::ROOT)
                        }
                        CanonicalizeMode::Response { .. } => {
                            CanonicalVarKind::Const(self.delegate.universe_of_const(vid).unwrap())
                        }
                    }
                }
                ty::InferConst::Fresh(_) => unimplemented!(),
            },
            ty::ConstKind::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => {
                    CanonicalVarKind::PlaceholderConst(PlaceholderConst::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
                CanonicalizeMode::Response { .. } => {
                    CanonicalVarKind::PlaceholderConst(placeholder)
                }
            },
            ty::ConstKind::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => {
                    CanonicalVarKind::PlaceholderConst(PlaceholderConst::new_anon(
                        ty::UniverseIndex::ROOT,
                        self.state.variables.len().into(),
                    ))
                }
                CanonicalizeMode::Response { .. } => panic!("param ty in response: {c:?}"),
            },
            // FIXME: See comment above -- we could fold the region separately or something.
            ty::ConstKind::Bound(_, _)
            | ty::ConstKind::Alias(_, _)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Error(_)
            | ty::ConstKind::Expr(_) => return c.super_fold_with(self),
        };

        let var = self.get_or_insert_bound_var(c, kind);

        Const::new_canonical_bound(self.cx(), var)
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate {
        if !p.flags().intersects(NEEDS_CANONICAL) { p } else { p.super_fold_with(self) }
    }

    fn fold_clauses(&mut self, c: I::Clauses) -> I::Clauses {
        match self.canonicalize_mode {
            CanonicalizeMode::Input(CanonicalizeInputKind::ParamEnv)
            | CanonicalizeMode::Response { max_input_universe: _ } => {}
            CanonicalizeMode::Input(CanonicalizeInputKind::Predicate) => {
                panic!("erasing 'static in env")
            }
        }
        if !c.flags().intersects(NEEDS_CANONICAL) { c } else { c.super_fold_with(self) }
    }
}
