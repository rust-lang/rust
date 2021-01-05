use rustc_middle::mir::interpret::InterpResult;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeVisitor};
use std::convert::TryInto;
use std::ops::ControlFlow;

/// Returns `true` if a used generic parameter requires substitution.
crate fn ensure_monomorphic_enough<'tcx, T>(tcx: TyCtxt<'tcx>, ty: T) -> InterpResult<'tcx>
where
    T: TypeFoldable<'tcx>,
{
    debug!("ensure_monomorphic_enough: ty={:?}", ty);
    if !ty.potentially_needs_subst() {
        return Ok(());
    }

    struct FoundParam;
    struct UsedParamsNeedSubstVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
    }

    impl<'tcx> TypeVisitor<'tcx> for UsedParamsNeedSubstVisitor<'tcx> {
        type BreakTy = FoundParam;

        fn tcx_for_anon_const_substs(&self) -> Option<TyCtxt<'tcx>> {
            Some(self.tcx)
        }

        fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            if !ty.potentially_needs_subst() {
                return ControlFlow::CONTINUE;
            }

            match *ty.kind() {
                ty::Param(_) => ControlFlow::Break(FoundParam),
                ty::Closure(def_id, substs)
                | ty::Generator(def_id, substs, ..)
                | ty::FnDef(def_id, substs) => {
                    let unused_params = self.tcx.unused_generic_params(def_id);
                    for (index, subst) in substs.into_iter().enumerate() {
                        let index = index
                            .try_into()
                            .expect("more generic parameters than can fit into a `u32`");
                        let is_used = unused_params.contains(index).map_or(true, |unused| !unused);
                        // Only recurse when generic parameters in fns, closures and generators
                        // are used and require substitution.
                        match (is_used, subst.definitely_needs_subst(self.tcx)) {
                            // Just in case there are closures or generators within this subst,
                            // recurse.
                            (true, true) => return subst.super_visit_with(self),
                            // Confirm that polymorphization replaced the parameter with
                            // `ty::Param`/`ty::ConstKind::Param`.
                            (false, true) if cfg!(debug_assertions) => match subst.unpack() {
                                ty::subst::GenericArgKind::Type(ty) => {
                                    assert!(matches!(ty.kind(), ty::Param(_)))
                                }
                                ty::subst::GenericArgKind::Const(ct) => {
                                    assert!(matches!(ct.val, ty::ConstKind::Param(_)))
                                }
                                ty::subst::GenericArgKind::Lifetime(..) => (),
                            },
                            _ => {}
                        }
                    }
                    ControlFlow::CONTINUE
                }
                _ => ty.super_visit_with(self),
            }
        }

        fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            match c.val {
                ty::ConstKind::Param(..) => ControlFlow::Break(FoundParam),
                _ => c.super_visit_with(self),
            }
        }
    }

    let mut vis = UsedParamsNeedSubstVisitor { tcx };
    if matches!(ty.visit_with(&mut vis), ControlFlow::Break(FoundParam)) {
        throw_inval!(TooGeneric);
    } else {
        Ok(())
    }
}
