use crate::ty::subst::{GenericArg, GenericArgKind};
use crate::ty::{self, InferConst, Ty, TypeFlags};
use std::slice;

#[derive(Debug)]
pub struct FlagComputation {
    pub flags: TypeFlags,

    // see `TyS::outer_exclusive_binder` for details
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

    pub fn for_predicate(kind: &ty::PredicateKind<'_>) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_predicate_kind(kind);
        result
    }

    pub fn for_const(c: &ty::Const<'_>) -> TypeFlags {
        let mut result = FlagComputation::new();
        result.add_const(c);
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
    fn add_bound_computation(&mut self, computation: FlagComputation) {
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
                self.add_substs(substs);
            }

            &ty::GeneratorWitness(ref ts) => {
                let mut computation = FlagComputation::new();
                computation.add_tys(&ts.skip_binder()[..]);
                self.add_bound_computation(computation);
            }

            &ty::Closure(_, ref substs) => {
                self.add_substs(substs);
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
                    ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {}

                    ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_) => {
                        self.add_flags(TypeFlags::HAS_TY_INFER)
                    }
                }
            }

            &ty::Adt(_, substs) => {
                self.add_substs(substs);
            }

            &ty::Projection(ref data) => {
                self.add_flags(TypeFlags::HAS_TY_PROJECTION);
                self.add_projection_ty(data);
            }

            &ty::Opaque(_, substs) => {
                self.add_flags(TypeFlags::HAS_TY_OPAQUE);
                self.add_substs(substs);
            }

            &ty::Dynamic(ref obj, r) => {
                let mut computation = FlagComputation::new();
                for predicate in obj.skip_binder().iter() {
                    match predicate {
                        ty::ExistentialPredicate::Trait(tr) => computation.add_substs(tr.substs),
                        ty::ExistentialPredicate::Projection(p) => {
                            let mut proj_computation = FlagComputation::new();
                            proj_computation.add_existential_projection(&p);
                            self.add_bound_computation(proj_computation);
                        }
                        ty::ExistentialPredicate::AutoTrait(_) => {}
                    }
                }
                self.add_bound_computation(computation);
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

            &ty::FnPtr(f) => {
                self.add_fn_sig(f);
            }
        }
    }

    fn add_predicate_kind(&mut self, kind: &ty::PredicateKind<'_>) {
        match kind {
            ty::PredicateKind::Trait(trait_pred, _constness) => {
                let mut computation = FlagComputation::new();
                computation.add_substs(trait_pred.skip_binder().trait_ref.substs);

                self.add_bound_computation(computation);
            }
            ty::PredicateKind::RegionOutlives(poly_outlives) => {
                let mut computation = FlagComputation::new();
                let ty::OutlivesPredicate(a, b) = poly_outlives.skip_binder();
                computation.add_region(a);
                computation.add_region(b);

                self.add_bound_computation(computation);
            }
            ty::PredicateKind::TypeOutlives(poly_outlives) => {
                let mut computation = FlagComputation::new();
                let ty::OutlivesPredicate(ty, region) = poly_outlives.skip_binder();
                computation.add_ty(ty);
                computation.add_region(region);

                self.add_bound_computation(computation);
            }
            ty::PredicateKind::Subtype(poly_subtype) => {
                let mut computation = FlagComputation::new();
                let ty::SubtypePredicate { a_is_expected: _, a, b } = poly_subtype.skip_binder();
                computation.add_ty(a);
                computation.add_ty(b);

                self.add_bound_computation(computation);
            }
            ty::PredicateKind::Projection(projection) => {
                let mut computation = FlagComputation::new();
                let ty::ProjectionPredicate { projection_ty, ty } = projection.skip_binder();
                computation.add_projection_ty(projection_ty);
                computation.add_ty(ty);

                self.add_bound_computation(computation);
            }
            ty::PredicateKind::WellFormed(arg) => {
                self.add_substs(slice::from_ref(arg));
            }
            ty::PredicateKind::ObjectSafe(_def_id) => {}
            ty::PredicateKind::ClosureKind(_def_id, substs, _kind) => {
                self.add_substs(substs);
            }
            ty::PredicateKind::ConstEvaluatable(_def_id, substs) => {
                self.add_substs(substs);
            }
            ty::PredicateKind::ConstEquate(expected, found) => {
                self.add_const(expected);
                self.add_const(found);
            }
        }
    }

    fn add_ty(&mut self, ty: Ty<'_>) {
        self.add_flags(ty.flags);
        self.add_exclusive_binder(ty.outer_exclusive_binder);
    }

    fn add_tys(&mut self, tys: &[Ty<'_>]) {
        for &ty in tys {
            self.add_ty(ty);
        }
    }

    fn add_fn_sig(&mut self, fn_sig: ty::PolyFnSig<'_>) {
        let mut computation = FlagComputation::new();

        computation.add_tys(fn_sig.skip_binder().inputs());
        computation.add_ty(fn_sig.skip_binder().output());

        self.add_bound_computation(computation);
    }

    fn add_region(&mut self, r: ty::Region<'_>) {
        self.add_flags(r.type_flags());
        if let ty::ReLateBound(debruijn, _) = *r {
            self.add_bound_var(debruijn);
        }
    }

    fn add_const(&mut self, c: &ty::Const<'_>) {
        self.add_ty(c.ty);
        match c.val {
            ty::ConstKind::Unevaluated(_, substs, _) => {
                self.add_substs(substs);
                self.add_flags(TypeFlags::HAS_CT_PROJECTION);
            }
            ty::ConstKind::Infer(infer) => {
                self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);
                match infer {
                    InferConst::Fresh(_) => {}
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

    fn add_existential_projection(&mut self, projection: &ty::ExistentialProjection<'_>) {
        self.add_substs(projection.substs);
        self.add_ty(projection.ty);
    }

    fn add_projection_ty(&mut self, projection_ty: &ty::ProjectionTy<'_>) {
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
