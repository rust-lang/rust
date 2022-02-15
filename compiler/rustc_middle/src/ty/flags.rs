use crate::ty::subst::{GenericArg, GenericArgKind};
use crate::ty::{self, InferConst, Term, Ty, TypeFlags};
use std::slice;

#[derive(Debug)]
pub struct FlagComputation {
    pub flags: TypeFlags,

    // see `Ty::outer_exclusive_binder` for details
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

    pub fn for_predicate<'tcx>(binder: ty::Binder<'tcx, ty::PredicateKind<'_>>) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_predicate(binder);
        result
    }

    pub fn for_const(c: ty::Const<'_>) -> TypeFlags {
        let mut result = FlagComputation::new();
        result.add_const(c);
        result.flags
    }

    pub fn for_unevaluated_const(uv: ty::Unevaluated<'_>) -> TypeFlags {
        let mut result = FlagComputation::new();
        result.add_unevaluated_const(uv);
        result.flags
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
            computation.flags = computation.flags | TypeFlags::HAS_RE_LATE_BOUND;
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
                self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
            }

            &ty::Generator(_, ref substs, _) => {
                let substs = substs.as_generator();
                let should_remove_further_specializable =
                    !self.flags.contains(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
                self.add_substs(substs.parent_substs());
                if should_remove_further_specializable {
                    self.flags -= TypeFlags::STILL_FURTHER_SPECIALIZABLE;
                }

                self.add_ty(substs.resume_ty());
                self.add_ty(substs.return_ty());
                self.add_ty(substs.witness());
                self.add_ty(substs.yield_ty());
                self.add_ty(substs.tupled_upvars_ty());
            }

            &ty::GeneratorWitness(ts) => {
                self.bound_computation(ts, |flags, ts| flags.add_tys(ts));
            }

            &ty::Closure(_, substs) => {
                let substs = substs.as_closure();
                let should_remove_further_specializable =
                    !self.flags.contains(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
                self.add_substs(substs.parent_substs());
                if should_remove_further_specializable {
                    self.flags -= TypeFlags::STILL_FURTHER_SPECIALIZABLE;
                }

                self.add_ty(substs.sig_as_fn_ptr_ty());
                self.add_ty(substs.kind_ty());
                self.add_ty(substs.tupled_upvars_ty());
            }

            &ty::Bound(debruijn, _) => {
                self.add_bound_var(debruijn);
            }

            &ty::Placeholder(..) => {
                self.add_flags(TypeFlags::HAS_TY_PLACEHOLDER);
                self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
            }

            &ty::Infer(infer) => {
                self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
                match infer {
                    ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {
                        self.add_flags(TypeFlags::HAS_TY_FRESH)
                    }

                    ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_) => {
                        self.add_flags(TypeFlags::HAS_TY_INFER)
                    }
                }
            }

            &ty::Adt(_, substs) => {
                self.add_substs(substs);
            }

            &ty::Projection(data) => {
                self.add_flags(TypeFlags::HAS_TY_PROJECTION);
                self.add_projection_ty(data);
            }

            &ty::Opaque(_, substs) => {
                self.add_flags(TypeFlags::HAS_TY_OPAQUE);
                self.add_substs(substs);
            }

            &ty::Dynamic(obj, r) => {
                for predicate in obj.iter() {
                    self.bound_computation(predicate, |computation, predicate| match predicate {
                        ty::ExistentialPredicate::Trait(tr) => computation.add_substs(tr.substs),
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

            &ty::Slice(tt) => self.add_ty(tt),

            &ty::RawPtr(ref m) => {
                self.add_ty(m.ty);
            }

            &ty::Ref(r, ty, _) => {
                self.add_region(r);
                self.add_ty(ty);
            }

            &ty::Tuple(ref substs) => {
                self.add_substs(substs);
            }

            &ty::FnDef(_, substs) => {
                self.add_substs(substs);
            }

            &ty::FnPtr(fn_sig) => self.bound_computation(fn_sig, |computation, fn_sig| {
                computation.add_tys(fn_sig.inputs());
                computation.add_ty(fn_sig.output());
            }),
        }
    }

    fn add_predicate(&mut self, binder: ty::Binder<'_, ty::PredicateKind<'_>>) {
        self.bound_computation(binder, |computation, atom| computation.add_predicate_atom(atom));
    }

    fn add_predicate_atom(&mut self, atom: ty::PredicateKind<'_>) {
        match atom {
            ty::PredicateKind::Trait(trait_pred) => {
                self.add_substs(trait_pred.trait_ref.substs);
            }
            ty::PredicateKind::RegionOutlives(ty::OutlivesPredicate(a, b)) => {
                self.add_region(a);
                self.add_region(b);
            }
            ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(ty, region)) => {
                self.add_ty(ty);
                self.add_region(region);
            }
            ty::PredicateKind::Subtype(ty::SubtypePredicate { a_is_expected: _, a, b }) => {
                self.add_ty(a);
                self.add_ty(b);
            }
            ty::PredicateKind::Coerce(ty::CoercePredicate { a, b }) => {
                self.add_ty(a);
                self.add_ty(b);
            }
            ty::PredicateKind::Projection(ty::ProjectionPredicate { projection_ty, term }) => {
                self.add_projection_ty(projection_ty);
                match term {
                    Term::Ty(ty) => self.add_ty(ty),
                    Term::Const(c) => self.add_const(c),
                }
            }
            ty::PredicateKind::WellFormed(arg) => {
                self.add_substs(slice::from_ref(&arg));
            }
            ty::PredicateKind::ObjectSafe(_def_id) => {}
            ty::PredicateKind::ClosureKind(_def_id, substs, _kind) => {
                self.add_substs(substs);
            }
            ty::PredicateKind::ConstEvaluatable(uv) => {
                self.add_unevaluated_const(uv);
            }
            ty::PredicateKind::ConstEquate(expected, found) => {
                self.add_const(expected);
                self.add_const(found);
            }
            ty::PredicateKind::TypeWellFormedFromEnv(ty) => {
                self.add_ty(ty);
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
        if let ty::ReLateBound(debruijn, _) = *r {
            self.add_bound_var(debruijn);
        }
    }

    fn add_const(&mut self, c: ty::Const<'_>) {
        self.add_ty(c.ty());
        match c.val() {
            ty::ConstKind::Unevaluated(unevaluated) => self.add_unevaluated_const(unevaluated),
            ty::ConstKind::Infer(infer) => {
                self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
                match infer {
                    InferConst::Fresh(_) => self.add_flags(TypeFlags::HAS_CT_FRESH),
                    InferConst::Var(_) => self.add_flags(TypeFlags::HAS_CT_INFER),
                }
            }
            ty::ConstKind::Bound(debruijn, _) => {
                self.add_bound_var(debruijn);
            }
            ty::ConstKind::Param(_) => {
                self.add_flags(TypeFlags::HAS_CT_PARAM);
                self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
            }
            ty::ConstKind::Placeholder(_) => {
                self.add_flags(TypeFlags::HAS_CT_PLACEHOLDER);
                self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
            }
            ty::ConstKind::Value(_) => {}
            ty::ConstKind::Error(_) => self.add_flags(TypeFlags::HAS_ERROR),
        }
    }

    fn add_unevaluated_const<P>(&mut self, ct: ty::Unevaluated<'_, P>) {
        self.add_substs(ct.substs);
        self.add_flags(TypeFlags::HAS_CT_PROJECTION);
    }

    fn add_existential_projection(&mut self, projection: &ty::ExistentialProjection<'_>) {
        self.add_substs(projection.substs);
        match projection.term {
            ty::Term::Ty(ty) => self.add_ty(ty),
            ty::Term::Const(ct) => self.add_const(ct),
        }
    }

    fn add_projection_ty(&mut self, projection_ty: ty::ProjectionTy<'_>) {
        self.add_substs(projection_ty.substs);
    }

    fn add_substs(&mut self, substs: &[GenericArg<'_>]) {
        for kind in substs {
            match kind.unpack() {
                GenericArgKind::Type(ty) => self.add_ty(ty),
                GenericArgKind::Lifetime(lt) => self.add_region(lt),
                GenericArgKind::Const(ct) => self.add_const(ct),
            }
        }
    }
}
