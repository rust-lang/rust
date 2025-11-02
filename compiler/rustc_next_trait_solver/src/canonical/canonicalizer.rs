use rustc_type_ir::data_structures::{HashMap, ensure_sufficient_stack};
use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::{Goal, QueryInput};
use rustc_type_ir::{
    self as ty, Canonical, CanonicalParamEnvCacheEntry, CanonicalVarKind, Flags, InferCtxtLike,
    Interner, TypeFlags, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
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
    variables: &'a mut Vec<I::GenericArg>,
    var_kinds: Vec<CanonicalVarKind<I>>,
    variable_lookup_table: HashMap<I::GenericArg, usize>,
    /// Maps each `sub_unification_table_root_var` to the index of the first
    /// variable which used it.
    ///
    /// This means in case two type variables have the same sub relations root,
    /// we set the `sub_root` of the second variable to the position of the first.
    /// Otherwise the `sub_root` of each type variable is just its own position.
    sub_root_lookup_table: HashMap<ty::TyVid, usize>,

    /// We can simply cache based on the ty itself, because we use
    /// `ty::BoundVarIndexKind::Canonical`.
    cache: HashMap<I::Ty, I::Ty>,
}

impl<'a, D: SolverDelegate<Interner = I>, I: Interner> Canonicalizer<'a, D, I> {
    pub(super) fn canonicalize_response<T: TypeFoldable<I>>(
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
            sub_root_lookup_table: Default::default(),
            var_kinds: Vec::new(),

            cache: Default::default(),
        };

        let value = if value.has_type_flags(NEEDS_CANONICAL) {
            value.fold_with(&mut canonicalizer)
        } else {
            value
        };
        debug_assert!(!value.has_infer(), "unexpected infer in {value:?}");
        debug_assert!(!value.has_placeholders(), "unexpected placeholders in {value:?}");
        let (max_universe, variables) = canonicalizer.finalize();
        Canonical { max_universe, variables, value }
    }
    fn canonicalize_param_env(
        delegate: &'a D,
        variables: &'a mut Vec<I::GenericArg>,
        param_env: I::ParamEnv,
    ) -> (I::ParamEnv, HashMap<I::GenericArg, usize>, Vec<CanonicalVarKind<I>>) {
        if !param_env.has_type_flags(NEEDS_CANONICAL) {
            return (param_env, Default::default(), Vec::new());
        }

        // Check whether we can use the global cache for this param_env. As we only use
        // the `param_env` itself as the cache key, considering any additional information
        // durnig its canonicalization would be incorrect. We always canonicalize region
        // inference variables in a separate universe, so these are fine. However, we do
        // track the universe of type and const inference variables so these must not be
        // globally cached. We don't rely on any additional information when canonicalizing
        // placeholders.
        if !param_env.has_non_region_infer() {
            delegate.cx().canonical_param_env_cache_get_or_insert(
                param_env,
                || {
                    let mut variables = Vec::new();
                    let mut env_canonicalizer = Canonicalizer {
                        delegate,
                        canonicalize_mode: CanonicalizeMode::Input(CanonicalizeInputKind::ParamEnv),

                        variables: &mut variables,
                        variable_lookup_table: Default::default(),
                        sub_root_lookup_table: Default::default(),
                        var_kinds: Vec::new(),

                        cache: Default::default(),
                    };
                    let param_env = param_env.fold_with(&mut env_canonicalizer);
                    debug_assert!(env_canonicalizer.sub_root_lookup_table.is_empty());
                    CanonicalParamEnvCacheEntry {
                        param_env,
                        variable_lookup_table: env_canonicalizer.variable_lookup_table,
                        var_kinds: env_canonicalizer.var_kinds,
                        variables,
                    }
                },
                |&CanonicalParamEnvCacheEntry {
                     param_env,
                     variables: ref cache_variables,
                     ref variable_lookup_table,
                     ref var_kinds,
                 }| {
                    debug_assert!(variables.is_empty());
                    variables.extend(cache_variables.iter().copied());
                    (param_env, variable_lookup_table.clone(), var_kinds.clone())
                },
            )
        } else {
            let mut env_canonicalizer = Canonicalizer {
                delegate,
                canonicalize_mode: CanonicalizeMode::Input(CanonicalizeInputKind::ParamEnv),

                variables,
                variable_lookup_table: Default::default(),
                sub_root_lookup_table: Default::default(),
                var_kinds: Vec::new(),

                cache: Default::default(),
            };
            let param_env = param_env.fold_with(&mut env_canonicalizer);
            debug_assert!(env_canonicalizer.sub_root_lookup_table.is_empty());
            (param_env, env_canonicalizer.variable_lookup_table, env_canonicalizer.var_kinds)
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
        variables: &'a mut Vec<I::GenericArg>,
        input: QueryInput<I, P>,
    ) -> ty::Canonical<I, QueryInput<I, P>> {
        // First canonicalize the `param_env` while keeping `'static`
        let (param_env, variable_lookup_table, var_kinds) =
            Canonicalizer::canonicalize_param_env(delegate, variables, input.goal.param_env);
        // Then canonicalize the rest of the input without keeping `'static`
        // while *mostly* reusing the canonicalizer from above.
        let mut rest_canonicalizer = Canonicalizer {
            delegate,
            canonicalize_mode: CanonicalizeMode::Input(CanonicalizeInputKind::Predicate),

            variables,
            variable_lookup_table,
            sub_root_lookup_table: Default::default(),
            var_kinds,

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

        debug_assert!(!value.has_infer(), "unexpected infer in {value:?}");
        debug_assert!(!value.has_placeholders(), "unexpected placeholders in {value:?}");
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

    fn get_or_insert_sub_root(&mut self, vid: ty::TyVid) -> ty::BoundVar {
        let root_vid = self.delegate.sub_unification_table_root_var(vid);
        let idx =
            *self.sub_root_lookup_table.entry(root_vid).or_insert_with(|| self.variables.len());
        ty::BoundVar::from(idx)
    }

    fn finalize(self) -> (ty::UniverseIndex, I::CanonicalVarKinds) {
        let mut var_kinds = self.var_kinds;
        // See the rustc-dev-guide section about how we deal with universes
        // during canonicalization in the new solver.
        match self.canonicalize_mode {
            // All placeholders and vars are canonicalized in the root universe.
            CanonicalizeMode::Input { .. } => {
                debug_assert!(
                    var_kinds.iter().all(|var| var.universe() == ty::UniverseIndex::ROOT),
                    "expected all vars to be canonicalized in root universe: {var_kinds:#?}"
                );
                let var_kinds = self.delegate.cx().mk_canonical_var_kinds(&var_kinds);
                (ty::UniverseIndex::ROOT, var_kinds)
            }
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
                (max_universe, var_kinds)
            }
        }
    }

    fn inner_fold_ty(&mut self, t: I::Ty) -> I::Ty {
        let kind = match t.kind() {
            ty::Infer(i) => match i {
                ty::TyVar(vid) => {
                    debug_assert_eq!(
                        self.delegate.opportunistic_resolve_ty_var(vid),
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
                        self.delegate.opportunistic_resolve_int_var(vid),
                        t,
                        "ty vid should have been resolved fully before canonicalization"
                    );
                    CanonicalVarKind::Int
                }
                ty::FloatVar(vid) => {
                    debug_assert_eq!(
                        self.delegate.opportunistic_resolve_float_var(vid),
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
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderTy(
                    PlaceholderLike::new_anon(ty::UniverseIndex::ROOT, self.variables.len().into()),
                ),
                CanonicalizeMode::Response { .. } => CanonicalVarKind::PlaceholderTy(placeholder),
            },
            ty::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderTy(
                    PlaceholderLike::new_anon(ty::UniverseIndex::ROOT, self.variables.len().into()),
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
                return if t.has_type_flags(NEEDS_CANONICAL) {
                    ensure_sufficient_stack(|| t.super_fold_with(self))
                } else {
                    t
                };
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

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        let kind = match r.kind() {
            ty::ReBound(..) => return r,

            // We don't canonicalize `ReStatic` in the `param_env` as we use it
            // when checking whether a `ParamEnv` candidate is global.
            ty::ReStatic => match self.canonicalize_mode {
                CanonicalizeMode::Input(CanonicalizeInputKind::Predicate { .. }) => {
                    CanonicalVarKind::Region(ty::UniverseIndex::ROOT)
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
                CanonicalizeMode::Input(_) => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => return r,
            },

            ty::ReEarlyParam(_) | ty::ReLateParam(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input(_) => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                CanonicalizeMode::Response { .. } => {
                    panic!("unexpected region in response: {r:?}")
                }
            },

            ty::RePlaceholder(placeholder) => match self.canonicalize_mode {
                // We canonicalize placeholder regions as existentials in query inputs.
                CanonicalizeMode::Input(_) => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
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
                debug_assert_eq!(
                    self.delegate.opportunistic_resolve_lt_var(vid),
                    r,
                    "region vid should have been resolved fully before canonicalization"
                );
                match self.canonicalize_mode {
                    CanonicalizeMode::Input(_) => CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
                    CanonicalizeMode::Response { .. } => {
                        CanonicalVarKind::Region(self.delegate.universe_of_lt(vid).unwrap())
                    }
                }
            }
        };

        let var = self.get_or_insert_bound_var(r, kind);

        Region::new_canonical_bound(self.cx(), var)
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        if let Some(&ty) = self.cache.get(&t) {
            ty
        } else {
            let res = self.inner_fold_ty(t);
            let old = self.cache.insert(t, res);
            assert_eq!(old, None);
            res
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        let kind = match c.kind() {
            ty::ConstKind::Infer(i) => match i {
                ty::InferConst::Var(vid) => {
                    debug_assert_eq!(
                        self.delegate.opportunistic_resolve_ct_var(vid),
                        c,
                        "const vid should have been resolved fully before canonicalization"
                    );

                    match self.canonicalize_mode {
                        CanonicalizeMode::Input { .. } => {
                            CanonicalVarKind::Const(ty::UniverseIndex::ROOT)
                        }
                        CanonicalizeMode::Response { .. } => {
                            CanonicalVarKind::Const(self.delegate.universe_of_ct(vid).unwrap())
                        }
                    }
                }
                ty::InferConst::Fresh(_) => todo!(),
            },
            ty::ConstKind::Placeholder(placeholder) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderConst(
                    PlaceholderLike::new_anon(ty::UniverseIndex::ROOT, self.variables.len().into()),
                ),
                CanonicalizeMode::Response { .. } => {
                    CanonicalVarKind::PlaceholderConst(placeholder)
                }
            },
            ty::ConstKind::Param(_) => match self.canonicalize_mode {
                CanonicalizeMode::Input { .. } => CanonicalVarKind::PlaceholderConst(
                    PlaceholderLike::new_anon(ty::UniverseIndex::ROOT, self.variables.len().into()),
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

        Const::new_canonical_bound(self.cx(), var)
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate {
        if p.flags().intersects(NEEDS_CANONICAL) { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: I::Clauses) -> I::Clauses {
        match self.canonicalize_mode {
            CanonicalizeMode::Input(CanonicalizeInputKind::ParamEnv)
            | CanonicalizeMode::Response { max_input_universe: _ } => {}
            CanonicalizeMode::Input(CanonicalizeInputKind::Predicate { .. }) => {
                panic!("erasing 'static in env")
            }
        }
        if c.flags().intersects(NEEDS_CANONICAL) { c.super_fold_with(self) } else { c }
    }
}
