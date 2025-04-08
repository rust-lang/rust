use std::slice;

use crate::ty::{self, GenericArg, GenericArgKind, InferConst, Ty, TypeFlags};

#[derive(Debug)]
pub struct FlagComputation {
    pub flags: TypeFlags,

    /// see `Ty::outer_exclusive_binder` for details
    pub outer_exclusive_binder: ty::DebruijnIndex,
}

impl FlagComputation {
    fn new() -> FlagComputation {
        FlagComputation { flags: TypeFlags::empty(), outer_exclusive_binder: ty::INNERMOST }
    }

    #[allow(rustc::usage_of_ty_tykind)]
    pub fn for_kind(kind: &ty::TyKind<'_>) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_kind(kind);
        result
    }

    pub fn for_predicate(binder: ty::Binder<'_, ty::PredicateKind<'_>>) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_predicate(binder);
        result
    }

    pub fn for_const_kind(kind: &ty::ConstKind<'_>) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_const_kind(kind);
        result
    }

    pub fn for_clauses(clauses: &[ty::Clause<'_>]) -> FlagComputation {
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
    fn bound_computation<T, F>(&mut self, value: ty::Binder<'_, T>, f: F)
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
    fn add_kind(&mut self, kind: &ty::TyKind<'_>) {
        match kind {
            &ty::Bool
            | &ty::Char
            | &ty::Int(_)
            | &ty::Float(_)
            | &ty::Uint(_)
            | &ty::Never
            | &ty::Str
            | &ty::Foreign(..) => {}

            &ty::Error(_) => self.add_flags(TypeFlags::HAS_ERROR),

            &ty::Param(_) => {
                self.add_flags(TypeFlags::HAS_TY_PARAM);
            }

            &ty::Closure(_, args)
            | &ty::Coroutine(_, args)
            | &ty::CoroutineClosure(_, args)
            | &ty::CoroutineWitness(_, args) => {
                self.add_args(args);
            }

            &ty::Bound(debruijn, _) => {
                self.add_bound_var(debruijn);
                self.add_flags(TypeFlags::HAS_TY_BOUND);
            }

            &ty::Placeholder(..) => {
                self.add_flags(TypeFlags::HAS_TY_PLACEHOLDER);
            }

            &ty::Infer(infer) => match infer {
                ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {
                    self.add_flags(TypeFlags::HAS_TY_FRESH)
                }

                ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_) => {
                    self.add_flags(TypeFlags::HAS_TY_INFER)
                }
            },

            &ty::Adt(_, args) => {
                self.add_args(args);
            }

            &ty::Alias(kind, data) => {
                self.add_flags(match kind {
                    ty::Projection => TypeFlags::HAS_TY_PROJECTION,
                    ty::Weak => TypeFlags::HAS_TY_WEAK,
                    ty::Opaque => TypeFlags::HAS_TY_OPAQUE,
                    ty::Inherent => TypeFlags::HAS_TY_INHERENT,
                });

                self.add_alias_ty(data);
            }

            &ty::Dynamic(obj, r, _) => {
                for predicate in obj.iter() {
                    self.bound_computation(predicate, |computation, predicate| match predicate {
                        ty::ExistentialPredicate::Trait(tr) => computation.add_args(tr.args),
                        ty::ExistentialPredicate::Projection(p) => {
                            computation.add_existential_projection(&p);
                        }
                        ty::ExistentialPredicate::AutoTrait(_) => {}
                    });
                }

                self.add_region(r);
            }

            &ty::Array(tt, len) => {
                self.add_ty(tt);
                self.add_const(len);
            }

            &ty::Pat(ty, pat) => {
                self.add_ty(ty);
                match *pat {
                    ty::PatternKind::Range { start, end } => {
                        self.add_const(start);
                        self.add_const(end);
                    }
                }
            }

            &ty::Slice(tt) => self.add_ty(tt),

            &ty::RawPtr(ty, _) => {
                self.add_ty(ty);
            }

            &ty::Ref(r, ty, _) => {
                self.add_region(r);
                self.add_ty(ty);
            }

            &ty::Tuple(types) => {
                self.add_tys(types);
            }

            &ty::FnDef(_, args) => {
                self.add_args(args);
            }

            &ty::FnPtr(sig_tys, _) => self.bound_computation(sig_tys, |computation, sig_tys| {
                computation.add_tys(sig_tys.inputs_and_output);
            }),

            &ty::UnsafeBinder(bound_ty) => {
                self.bound_computation(bound_ty.into(), |computation, ty| {
                    computation.add_ty(ty);
                })
            }
        }
    }

    fn add_predicate(&mut self, binder: ty::Binder<'_, ty::PredicateKind<'_>>) {
        self.bound_computation(binder, |computation, atom| computation.add_predicate_atom(atom));
    }

    fn add_predicate_atom(&mut self, atom: ty::PredicateKind<'_>) {
        match atom {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)) => {
                self.add_args(trait_pred.trait_ref.args);
            }
            ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(ty::HostEffectPredicate {
                trait_ref,
                constness: _,
            })) => {
                self.add_args(trait_ref.args);
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
            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) => {
                self.add_args(slice::from_ref(&arg));
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

    fn add_ty(&mut self, ty: Ty<'_>) {
        self.add_flags(ty.flags());
        self.add_exclusive_binder(ty.outer_exclusive_binder());
    }

    fn add_tys(&mut self, tys: &[Ty<'_>]) {
        for &ty in tys {
            self.add_ty(ty);
        }
    }

    fn add_region(&mut self, r: ty::Region<'_>) {
        self.add_flags(r.type_flags());
        if let ty::ReBound(debruijn, _) = r.kind() {
            self.add_bound_var(debruijn);
        }
    }

    fn add_const(&mut self, c: ty::Const<'_>) {
        self.add_flags(c.flags());
        self.add_exclusive_binder(c.outer_exclusive_binder());
    }

    fn add_const_kind(&mut self, c: &ty::ConstKind<'_>) {
        match *c {
            ty::ConstKind::Unevaluated(uv) => {
                self.add_args(uv.args);
                self.add_flags(TypeFlags::HAS_CT_PROJECTION);
            }
            ty::ConstKind::Infer(infer) => match infer {
                InferConst::Fresh(_) => self.add_flags(TypeFlags::HAS_CT_FRESH),
                InferConst::Var(_) => self.add_flags(TypeFlags::HAS_CT_INFER),
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
            ty::ConstKind::Value(cv) => self.add_ty(cv.ty),
            ty::ConstKind::Expr(e) => self.add_args(e.args()),
            ty::ConstKind::Error(_) => self.add_flags(TypeFlags::HAS_ERROR),
        }
    }

    fn add_existential_projection(&mut self, projection: &ty::ExistentialProjection<'_>) {
        self.add_args(projection.args);
        match projection.term.unpack() {
            ty::TermKind::Ty(ty) => self.add_ty(ty),
            ty::TermKind::Const(ct) => self.add_const(ct),
        }
    }

    fn add_alias_ty(&mut self, alias_ty: ty::AliasTy<'_>) {
        self.add_args(alias_ty.args);
    }

    fn add_alias_term(&mut self, alias_term: ty::AliasTerm<'_>) {
        self.add_args(alias_term.args);
    }

    fn add_args(&mut self, args: &[GenericArg<'_>]) {
        for kind in args {
            match kind.unpack() {
                GenericArgKind::Type(ty) => self.add_ty(ty),
                GenericArgKind::Lifetime(lt) => self.add_region(lt),
                GenericArgKind::Const(ct) => self.add_const(ct),
            }
        }
    }

    fn add_term(&mut self, term: ty::Term<'_>) {
        match term.unpack() {
            ty::TermKind::Ty(ty) => self.add_ty(ty),
            ty::TermKind::Const(ct) => self.add_const(ct),
        }
    }
}
