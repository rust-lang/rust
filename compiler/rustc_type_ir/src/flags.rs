use crate::inherent::*;
use crate::visit::Flags;
use crate::{self as ty, Interner};

bitflags::bitflags! {
    /// Flags that we track on types. These flags are propagated upwards
    /// through the type during type construction, so that we can quickly check
    /// whether the type has various kinds of types in it without recursing
    /// over the type itself.
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub struct TypeFlags: u32 {
        // Does this have parameters? Used to determine whether instantiation is
        // required.
        /// Does this have `Param`?
        const HAS_TY_PARAM                = 1 << 0;
        /// Does this have `ReEarlyParam`?
        const HAS_RE_PARAM                = 1 << 1;
        /// Does this have `ConstKind::Param`?
        const HAS_CT_PARAM                = 1 << 2;

        const HAS_PARAM                   = TypeFlags::HAS_TY_PARAM.bits()
                                          | TypeFlags::HAS_RE_PARAM.bits()
                                          | TypeFlags::HAS_CT_PARAM.bits();

        /// Does this have `Infer`?
        const HAS_TY_INFER                = 1 << 3;
        /// Does this have `ReVar`?
        const HAS_RE_INFER                = 1 << 4;
        /// Does this have `ConstKind::Infer`?
        const HAS_CT_INFER                = 1 << 5;

        /// Does this have inference variables? Used to determine whether
        /// inference is required.
        const HAS_INFER                   = TypeFlags::HAS_TY_INFER.bits()
                                          | TypeFlags::HAS_RE_INFER.bits()
                                          | TypeFlags::HAS_CT_INFER.bits();

        /// Does this have `Placeholder`?
        const HAS_TY_PLACEHOLDER          = 1 << 6;
        /// Does this have `RePlaceholder`?
        const HAS_RE_PLACEHOLDER          = 1 << 7;
        /// Does this have `ConstKind::Placeholder`?
        const HAS_CT_PLACEHOLDER          = 1 << 8;

        /// Does this have placeholders?
        const HAS_PLACEHOLDER             = TypeFlags::HAS_TY_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_RE_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits();

        /// `true` if there are "names" of regions and so forth
        /// that are local to a particular fn/inferctxt
        const HAS_FREE_LOCAL_REGIONS      = 1 << 9;

        /// `true` if there are "names" of types and regions and so forth
        /// that are local to a particular fn
        const HAS_FREE_LOCAL_NAMES        = TypeFlags::HAS_TY_PARAM.bits()
                                          | TypeFlags::HAS_CT_PARAM.bits()
                                          | TypeFlags::HAS_TY_INFER.bits()
                                          | TypeFlags::HAS_CT_INFER.bits()
                                          | TypeFlags::HAS_TY_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits()
                                          // We consider 'freshened' types and constants
                                          // to depend on a particular fn.
                                          // The freshening process throws away information,
                                          // which can make things unsuitable for use in a global
                                          // cache. Note that there is no 'fresh lifetime' flag -
                                          // freshening replaces all lifetimes with `ReErased`,
                                          // which is different from how types/const are freshened.
                                          | TypeFlags::HAS_TY_FRESH.bits()
                                          | TypeFlags::HAS_CT_FRESH.bits()
                                          | TypeFlags::HAS_FREE_LOCAL_REGIONS.bits()
                                          | TypeFlags::HAS_RE_ERASED.bits();

        /// Does this have `Projection`?
        const HAS_TY_PROJECTION           = 1 << 10;
        /// Does this have `Free` aliases?
        const HAS_TY_FREE_ALIAS                 = 1 << 11;
        /// Does this have `Opaque`?
        const HAS_TY_OPAQUE               = 1 << 12;
        /// Does this have `Inherent`?
        const HAS_TY_INHERENT             = 1 << 13;
        /// Does this have `ConstKind::Unevaluated`?
        const HAS_CT_PROJECTION           = 1 << 14;

        /// Does this have `Alias` or `ConstKind::Unevaluated`?
        ///
        /// Rephrased, could this term be normalized further?
        const HAS_ALIAS                   = TypeFlags::HAS_TY_PROJECTION.bits()
                                          | TypeFlags::HAS_TY_FREE_ALIAS.bits()
                                          | TypeFlags::HAS_TY_OPAQUE.bits()
                                          | TypeFlags::HAS_TY_INHERENT.bits()
                                          | TypeFlags::HAS_CT_PROJECTION.bits();

        /// Is an error type/lifetime/const reachable?
        const HAS_ERROR                   = 1 << 15;

        /// Does this have any region that "appears free" in the type?
        /// Basically anything but `ReBound` and `ReErased`.
        const HAS_FREE_REGIONS            = 1 << 16;

        /// Does this have any `ReBound` regions?
        const HAS_RE_BOUND                = 1 << 17;
        /// Does this have any `Bound` types?
        const HAS_TY_BOUND                = 1 << 18;
        /// Does this have any `ConstKind::Bound` consts?
        const HAS_CT_BOUND                = 1 << 19;
        /// Does this have any bound variables?
        /// Used to check if a global bound is safe to evaluate.
        const HAS_BOUND_VARS              = TypeFlags::HAS_RE_BOUND.bits()
                                          | TypeFlags::HAS_TY_BOUND.bits()
                                          | TypeFlags::HAS_CT_BOUND.bits();

        /// Does this have any `ReErased` regions?
        const HAS_RE_ERASED               = 1 << 20;

        /// Does this value have parameters/placeholders/inference variables which could be
        /// replaced later, in a way that would change the results of `impl` specialization?
        const STILL_FURTHER_SPECIALIZABLE = TypeFlags::HAS_TY_PARAM.bits()
                                          | TypeFlags::HAS_TY_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_TY_INFER.bits()
                                          | TypeFlags::HAS_CT_PARAM.bits()
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_CT_INFER.bits();

        /// Does this value have `InferTy::FreshTy/FreshIntTy/FreshFloatTy`?
        const HAS_TY_FRESH                = 1 << 21;

        /// Does this value have `InferConst::Fresh`?
        const HAS_CT_FRESH                = 1 << 22;

        /// Does this have any binders with bound vars (e.g. that need to be anonymized)?
        const HAS_BINDER_VARS             = 1 << 23;
    }
}

#[derive(Debug)]
pub struct FlagComputation<I> {
    pub flags: TypeFlags,

    /// see `Ty::outer_exclusive_binder` for details
    pub outer_exclusive_binder: ty::DebruijnIndex,

    interner: std::marker::PhantomData<I>,
}

impl<I: Interner> FlagComputation<I> {
    fn new() -> FlagComputation<I> {
        FlagComputation {
            flags: TypeFlags::empty(),
            outer_exclusive_binder: ty::INNERMOST,
            interner: std::marker::PhantomData,
        }
    }

    #[allow(rustc::usage_of_ty_tykind)]
    pub fn for_kind(kind: &ty::TyKind<I>) -> FlagComputation<I> {
        let mut result = FlagComputation::new();
        result.add_kind(kind);
        result
    }

    pub fn for_predicate(binder: ty::Binder<I, ty::PredicateKind<I>>) -> FlagComputation<I> {
        let mut result = FlagComputation::new();
        result.add_predicate(binder);
        result
    }

    pub fn for_const_kind(kind: &ty::ConstKind<I>) -> FlagComputation<I> {
        let mut result = FlagComputation::new();
        result.add_const_kind(kind);
        result
    }

    pub fn for_clauses(clauses: &[I::Clause]) -> FlagComputation<I> {
        let mut result = FlagComputation::new();
        for c in clauses {
            result.add_flags(c.as_predicate().flags());
            result.add_exclusive_binder(c.as_predicate().outer_exclusive_binder());
        }
        result
    }

    fn add_flags(&mut self, flags: TypeFlags) {
        self.flags = self.flags | flags;
    }

    /// indicates that `self` refers to something at binding level `binder`
    fn add_bound_var(&mut self, binder: ty::DebruijnIndex) {
        let exclusive_binder = binder.shifted_in(1);
        self.add_exclusive_binder(exclusive_binder);
    }

    /// indicates that `self` refers to something *inside* binding
    /// level `binder` -- not bound by `binder`, but bound by the next
    /// binder internal to it
    fn add_exclusive_binder(&mut self, exclusive_binder: ty::DebruijnIndex) {
        self.outer_exclusive_binder = self.outer_exclusive_binder.max(exclusive_binder);
    }

    /// Adds the flags/depth from a set of types that appear within the current type, but within a
    /// region binder.
    fn bound_computation<T, F>(&mut self, value: ty::Binder<I, T>, f: F)
    where
        F: FnOnce(&mut Self, T),
    {
        let mut computation = FlagComputation::new();

        if !value.bound_vars().is_empty() {
            computation.add_flags(TypeFlags::HAS_BINDER_VARS);
        }

        f(&mut computation, value.skip_binder());

        self.add_flags(computation.flags);

        // The types that contributed to `computation` occurred within
        // a region binder, so subtract one from the region depth
        // within when adding the depth to `self`.
        let outer_exclusive_binder = computation.outer_exclusive_binder;
        if outer_exclusive_binder > ty::INNERMOST {
            self.add_exclusive_binder(outer_exclusive_binder.shifted_out(1));
        } // otherwise, this binder captures nothing
    }

    #[allow(rustc::usage_of_ty_tykind)]
    fn add_kind(&mut self, kind: &ty::TyKind<I>) {
        match *kind {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Float(_)
            | ty::Uint(_)
            | ty::Never
            | ty::Str
            | ty::Foreign(..) => {}

            ty::Error(_) => self.add_flags(TypeFlags::HAS_ERROR),

            ty::Param(_) => {
                self.add_flags(TypeFlags::HAS_TY_PARAM);
            }

            ty::Closure(_, args)
            | ty::Coroutine(_, args)
            | ty::CoroutineClosure(_, args)
            | ty::CoroutineWitness(_, args) => {
                self.add_args(args.as_slice());
            }

            ty::Bound(debruijn, _) => {
                self.add_bound_var(debruijn);
                self.add_flags(TypeFlags::HAS_TY_BOUND);
            }

            ty::Placeholder(..) => {
                self.add_flags(TypeFlags::HAS_TY_PLACEHOLDER);
            }

            ty::Infer(infer) => match infer {
                ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {
                    self.add_flags(TypeFlags::HAS_TY_FRESH)
                }

                ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_) => {
                    self.add_flags(TypeFlags::HAS_TY_INFER)
                }
            },

            ty::Adt(_, args) => {
                self.add_args(args.as_slice());
            }

            ty::Alias(kind, data) => {
                self.add_flags(match kind {
                    ty::Projection => TypeFlags::HAS_TY_PROJECTION,
                    ty::Free => TypeFlags::HAS_TY_FREE_ALIAS,
                    ty::Opaque => TypeFlags::HAS_TY_OPAQUE,
                    ty::Inherent => TypeFlags::HAS_TY_INHERENT,
                });

                self.add_alias_ty(data);
            }

            ty::Dynamic(obj, r, _) => {
                for predicate in obj.iter() {
                    self.bound_computation(predicate, |computation, predicate| match predicate {
                        ty::ExistentialPredicate::Trait(tr) => {
                            computation.add_args(tr.args.as_slice())
                        }
                        ty::ExistentialPredicate::Projection(p) => {
                            computation.add_existential_projection(&p);
                        }
                        ty::ExistentialPredicate::AutoTrait(_) => {}
                    });
                }

                self.add_region(r);
            }

            ty::Array(tt, len) => {
                self.add_ty(tt);
                self.add_const(len);
            }

            ty::Pat(ty, pat) => {
                self.add_ty(ty);
                self.add_ty_pat(pat);
            }

            ty::Slice(tt) => self.add_ty(tt),

            ty::RawPtr(ty, _) => {
                self.add_ty(ty);
            }

            ty::Ref(r, ty, _) => {
                self.add_region(r);
                self.add_ty(ty);
            }

            ty::Tuple(types) => {
                self.add_tys(types);
            }

            ty::FnDef(_, args) => {
                self.add_args(args.as_slice());
            }

            ty::FnPtr(sig_tys, _) => self.bound_computation(sig_tys, |computation, sig_tys| {
                computation.add_tys(sig_tys.inputs_and_output);
            }),

            ty::UnsafeBinder(bound_ty) => {
                self.bound_computation(bound_ty.into(), |computation, ty| {
                    computation.add_ty(ty);
                })
            }
        }
    }

    fn add_ty_pat(&mut self, pat: <I as Interner>::Pat) {
        self.add_flags(pat.flags());
    }

    fn add_predicate(&mut self, binder: ty::Binder<I, ty::PredicateKind<I>>) {
        self.bound_computation(binder, |computation, atom| computation.add_predicate_atom(atom));
    }

    fn add_predicate_atom(&mut self, atom: ty::PredicateKind<I>) {
        match atom {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)) => {
                self.add_args(trait_pred.trait_ref.args.as_slice());
            }
            ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(ty::HostEffectPredicate {
                trait_ref,
                constness: _,
            })) => {
                self.add_args(trait_ref.args.as_slice());
            }
            ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(
                a,
                b,
            ))) => {
                self.add_region(a);
                self.add_region(b);
            }
            ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
                ty,
                region,
            ))) => {
                self.add_ty(ty);
                self.add_region(region);
            }
            ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ty)) => {
                self.add_const(ct);
                self.add_ty(ty);
            }
            ty::PredicateKind::Subtype(ty::SubtypePredicate { a_is_expected: _, a, b }) => {
                self.add_ty(a);
                self.add_ty(b);
            }
            ty::PredicateKind::Coerce(ty::CoercePredicate { a, b }) => {
                self.add_ty(a);
                self.add_ty(b);
            }
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(ty::ProjectionPredicate {
                projection_term,
                term,
            })) => {
                self.add_alias_term(projection_term);
                self.add_term(term);
            }
            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                self.add_term(term);
            }
            ty::PredicateKind::DynCompatible(_def_id) => {}
            ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(uv)) => {
                self.add_const(uv);
            }
            ty::PredicateKind::ConstEquate(expected, found) => {
                self.add_const(expected);
                self.add_const(found);
            }
            ty::PredicateKind::Ambiguous => {}
            ty::PredicateKind::NormalizesTo(ty::NormalizesTo { alias, term }) => {
                self.add_alias_term(alias);
                self.add_term(term);
            }
            ty::PredicateKind::AliasRelate(t1, t2, _) => {
                self.add_term(t1);
                self.add_term(t2);
            }
        }
    }

    fn add_ty(&mut self, ty: I::Ty) {
        self.add_flags(ty.flags());
        self.add_exclusive_binder(ty.outer_exclusive_binder());
    }

    fn add_tys(&mut self, tys: I::Tys) {
        for ty in tys.iter() {
            self.add_ty(ty);
        }
    }

    fn add_region(&mut self, r: I::Region) {
        self.add_flags(r.flags());
        if let ty::ReBound(debruijn, _) = r.kind() {
            self.add_bound_var(debruijn);
        }
    }

    fn add_const(&mut self, c: I::Const) {
        self.add_flags(c.flags());
        self.add_exclusive_binder(c.outer_exclusive_binder());
    }

    fn add_const_kind(&mut self, c: &ty::ConstKind<I>) {
        match *c {
            ty::ConstKind::Unevaluated(uv) => {
                self.add_args(uv.args.as_slice());
                self.add_flags(TypeFlags::HAS_CT_PROJECTION);
            }
            ty::ConstKind::Infer(infer) => match infer {
                ty::InferConst::Fresh(_) => self.add_flags(TypeFlags::HAS_CT_FRESH),
                ty::InferConst::Var(_) => self.add_flags(TypeFlags::HAS_CT_INFER),
            },
            ty::ConstKind::Bound(debruijn, _) => {
                self.add_bound_var(debruijn);
                self.add_flags(TypeFlags::HAS_CT_BOUND);
            }
            ty::ConstKind::Param(_) => {
                self.add_flags(TypeFlags::HAS_CT_PARAM);
            }
            ty::ConstKind::Placeholder(_) => {
                self.add_flags(TypeFlags::HAS_CT_PLACEHOLDER);
            }
            ty::ConstKind::Value(cv) => self.add_ty(cv.ty()),
            ty::ConstKind::Expr(e) => self.add_args(e.args().as_slice()),
            ty::ConstKind::Error(_) => self.add_flags(TypeFlags::HAS_ERROR),
        }
    }

    fn add_existential_projection(&mut self, projection: &ty::ExistentialProjection<I>) {
        self.add_args(projection.args.as_slice());
        match projection.term.kind() {
            ty::TermKind::Ty(ty) => self.add_ty(ty),
            ty::TermKind::Const(ct) => self.add_const(ct),
        }
    }

    fn add_alias_ty(&mut self, alias_ty: ty::AliasTy<I>) {
        self.add_args(alias_ty.args.as_slice());
    }

    fn add_alias_term(&mut self, alias_term: ty::AliasTerm<I>) {
        self.add_args(alias_term.args.as_slice());
    }

    fn add_args(&mut self, args: &[I::GenericArg]) {
        for arg in args {
            match arg.kind() {
                ty::GenericArgKind::Type(ty) => self.add_ty(ty),
                ty::GenericArgKind::Lifetime(lt) => self.add_region(lt),
                ty::GenericArgKind::Const(ct) => self.add_const(ct),
            }
        }
    }

    fn add_term(&mut self, term: I::Term) {
        match term.kind() {
            ty::TermKind::Ty(ty) => self.add_ty(ty),
            ty::TermKind::Const(ct) => self.add_const(ct),
        }
    }
}
