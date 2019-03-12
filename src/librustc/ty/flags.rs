use crate::ty::subst::{SubstsRef, UnpackedKind};
use crate::ty::{self, Ty, TypeFlags, TypeFoldable, InferConst};
use crate::mir::interpret::ConstValue;

#[derive(Debug)]
pub struct FlagComputation {
    pub flags: TypeFlags,

    // see `TyS::outer_exclusive_binder` for details
    pub outer_exclusive_binder: ty::DebruijnIndex,
}

impl FlagComputation {
    fn new() -> FlagComputation {
        FlagComputation {
            flags: TypeFlags::empty(),
            outer_exclusive_binder: ty::INNERMOST,
        }
    }

    pub fn for_sty(st: &ty::TyKind<'_>) -> FlagComputation {
        let mut result = FlagComputation::new();
        result.add_sty(st);
        result
    }

    pub fn for_const(c: &ty::Const<'_>) -> TypeFlags {
        let mut result = FlagComputation::new();
        result.add_const(c);
        result.flags
    }

    fn add_flags(&mut self, flags: TypeFlags) {
        self.flags = self.flags | (flags & TypeFlags::NOMINAL_FLAGS);
    }

    /// indicates that `self` refers to something at binding level `binder`
    fn add_binder(&mut self, binder: ty::DebruijnIndex) {
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
    fn add_bound_computation(&mut self, computation: &FlagComputation) {
        self.add_flags(computation.flags);

        // The types that contributed to `computation` occurred within
        // a region binder, so subtract one from the region depth
        // within when adding the depth to `self`.
        let outer_exclusive_binder = computation.outer_exclusive_binder;
        if outer_exclusive_binder > ty::INNERMOST {
            self.add_exclusive_binder(outer_exclusive_binder.shifted_out(1));
        } // otherwise, this binder captures nothing
    }

    fn add_sty(&mut self, st: &ty::TyKind<'_>) {
        match st {
            &ty::Bool |
            &ty::Char |
            &ty::Int(_) |
            &ty::Float(_) |
            &ty::Uint(_) |
            &ty::Never |
            &ty::Str |
            &ty::Foreign(..) => {
            }

            // You might think that we could just return Error for
            // any type containing Error as a component, and get
            // rid of the TypeFlags::HAS_TY_ERR flag -- likewise for ty_bot (with
            // the exception of function types that return bot).
            // But doing so caused sporadic memory corruption, and
            // neither I (tjc) nor nmatsakis could figure out why,
            // so we're doing it this way.
            &ty::Error => {
                self.add_flags(TypeFlags::HAS_TY_ERR)
            }

            &ty::Param(ref p) => {
                self.add_flags(TypeFlags::HAS_FREE_LOCAL_NAMES);
                if p.is_self() {
                    self.add_flags(TypeFlags::HAS_SELF);
                } else {
                    self.add_flags(TypeFlags::HAS_PARAMS);
                }
            }

            &ty::Generator(_, ref substs, _) => {
                self.add_flags(TypeFlags::HAS_TY_CLOSURE);
                self.add_flags(TypeFlags::HAS_FREE_LOCAL_NAMES);
                self.add_substs(&substs.substs);
            }

            &ty::GeneratorWitness(ref ts) => {
                let mut computation = FlagComputation::new();
                computation.add_tys(&ts.skip_binder()[..]);
                self.add_bound_computation(&computation);
            }

            &ty::Closure(_, ref substs) => {
                self.add_flags(TypeFlags::HAS_TY_CLOSURE);
                self.add_flags(TypeFlags::HAS_FREE_LOCAL_NAMES);
                self.add_substs(&substs.substs);
            }

            &ty::Bound(debruijn, _) => {
                self.add_binder(debruijn);
            }

            &ty::Placeholder(..) => {
                self.add_flags(TypeFlags::HAS_TY_PLACEHOLDER);
            }

            &ty::Infer(infer) => {
                self.add_flags(TypeFlags::HAS_FREE_LOCAL_NAMES); // it might, right?
                self.add_flags(TypeFlags::HAS_TY_INFER);
                match infer {
                    ty::FreshTy(_) |
                    ty::FreshIntTy(_) |
                    ty::FreshFloatTy(_) => {
                    }

                    ty::TyVar(_) |
                    ty::IntVar(_) |
                    ty::FloatVar(_) => {
                        self.add_flags(TypeFlags::KEEP_IN_LOCAL_TCX)
                    }
                }
            }

            &ty::Adt(_, substs) => {
                self.add_substs(substs);
            }

            &ty::Projection(ref data) => {
                // currently we can't normalize projections that
                // include bound regions, so track those separately.
                if !data.has_escaping_bound_vars() {
                    self.add_flags(TypeFlags::HAS_NORMALIZABLE_PROJECTION);
                }
                self.add_flags(TypeFlags::HAS_PROJECTION);
                self.add_projection_ty(data);
            }

            &ty::UnnormalizedProjection(ref data) => {
                self.add_flags(TypeFlags::HAS_PROJECTION);
                self.add_projection_ty(data);
            },

            &ty::Opaque(_, substs) => {
                self.add_flags(TypeFlags::HAS_PROJECTION);
                self.add_substs(substs);
            }

            &ty::Dynamic(ref obj, r) => {
                let mut computation = FlagComputation::new();
                for predicate in obj.skip_binder().iter() {
                    match *predicate {
                        ty::ExistentialPredicate::Trait(tr) => computation.add_substs(tr.substs),
                        ty::ExistentialPredicate::Projection(p) => {
                            let mut proj_computation = FlagComputation::new();
                            proj_computation.add_existential_projection(&p);
                            self.add_bound_computation(&proj_computation);
                        }
                        ty::ExistentialPredicate::AutoTrait(_) => {}
                    }
                }
                self.add_bound_computation(&computation);
                self.add_region(r);
            }

            &ty::Array(tt, len) => {
                self.add_ty(tt);
                self.add_const(len);
            }

            &ty::Slice(tt) => {
                self.add_ty(tt)
            }

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

        self.add_bound_computation(&computation);
    }

    fn add_region(&mut self, r: ty::Region<'_>) {
        self.add_flags(r.type_flags());
        if let ty::ReLateBound(debruijn, _) = *r {
            self.add_binder(debruijn);
        }
    }

    fn add_const(&mut self, c: &ty::Const<'_>) {
        self.add_ty(c.ty);
        match c.val {
            ConstValue::Unevaluated(_, substs) => {
                self.add_substs(substs);
                self.add_flags(TypeFlags::HAS_NORMALIZABLE_PROJECTION | TypeFlags::HAS_PROJECTION);
            },
            ConstValue::Infer(infer) => {
                self.add_flags(TypeFlags::HAS_FREE_LOCAL_NAMES | TypeFlags::HAS_CT_INFER);
                match infer {
                    InferConst::Fresh(_) => {}
                    InferConst::Canonical(debruijn, _) => self.add_binder(debruijn),
                    InferConst::Var(_) => self.add_flags(TypeFlags::KEEP_IN_LOCAL_TCX),
                }
            }
            ConstValue::Param(_) => {
                self.add_flags(TypeFlags::HAS_FREE_LOCAL_NAMES | TypeFlags::HAS_PARAMS);
            }
            ConstValue::Placeholder(_) => {
                self.add_flags(TypeFlags::HAS_FREE_REGIONS | TypeFlags::HAS_CT_PLACEHOLDER);
            }
            _ => {},
        }
    }

    fn add_existential_projection(&mut self, projection: &ty::ExistentialProjection<'_>) {
        self.add_substs(projection.substs);
        self.add_ty(projection.ty);
    }

    fn add_projection_ty(&mut self, projection_ty: &ty::ProjectionTy<'_>) {
        self.add_substs(projection_ty.substs);
    }

    fn add_substs(&mut self, substs: SubstsRef<'_>) {
        for kind in substs {
            match kind.unpack() {
                UnpackedKind::Type(ty) => self.add_ty(ty),
                UnpackedKind::Lifetime(lt) => self.add_region(lt),
                UnpackedKind::Const(ct) => self.add_const(ct),
            }
        }
    }
}
