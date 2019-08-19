//! Utilities for creating generic types with bound vars in place of parameter values.

use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::{Kind, SubstsRef, InternalSubsts};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc_target::spec::abi;

crate fn bound(tcx: TyCtxt<'tcx>, index: u32) -> Ty<'tcx> {
    let ty = ty::Bound(
        ty::INNERMOST,
        ty::BoundVar::from_u32(index).into()
    );
    tcx.mk_ty(ty)
}

crate fn raw_ptr(tcx: TyCtxt<'tcx>, mutbl: hir::Mutability) -> Ty<'tcx> {
    tcx.mk_ptr(ty::TypeAndMut {
        ty: bound(tcx, 0),
        mutbl,
    })
}

crate fn fn_ptr(
    tcx: TyCtxt<'tcx>,
    arity_and_output: usize,
    c_variadic: bool,
    unsafety: hir::Unsafety,
    abi: abi::Abi,
) -> Ty<'tcx> {
    let inputs_and_output = tcx.mk_type_list(
        (0..arity_and_output).into_iter()
            .map(|i| ty::BoundVar::from(i))
            // DebruijnIndex(1) because we are going to inject these in a `PolyFnSig`
            .map(|var| tcx.mk_ty(ty::Bound(ty::DebruijnIndex::from(1usize), var.into())))
    );

    let fn_sig = ty::Binder::bind(ty::FnSig {
        inputs_and_output,
        c_variadic,
        unsafety,
        abi,
    });
    tcx.mk_fn_ptr(fn_sig)
}

crate fn type_list(tcx: TyCtxt<'tcx>, arity: usize) -> SubstsRef<'tcx> {
    tcx.mk_substs(
        (0..arity).into_iter()
            .map(|i| ty::BoundVar::from(i))
            .map(|var| tcx.mk_ty(ty::Bound(ty::INNERMOST, var.into())))
            .map(|ty| Kind::from(ty))
    )
}

crate fn ref_ty(tcx: TyCtxt<'tcx>, mutbl: hir::Mutability) -> Ty<'tcx> {
    let region = tcx.mk_region(
        ty::ReLateBound(ty::INNERMOST, ty::BoundRegion::BrAnon(0))
    );

    tcx.mk_ref(region, ty::TypeAndMut {
        ty: bound(tcx, 1),
        mutbl,
    })
}

crate fn fn_def(tcx: TyCtxt<'tcx>, def_id: DefId) -> Ty<'tcx> {
    tcx.mk_ty(ty::FnDef(def_id, InternalSubsts::bound_vars_for_item(tcx, def_id)))
}

crate fn closure(tcx: TyCtxt<'tcx>, def_id: DefId) -> Ty<'tcx> {
    tcx.mk_closure(def_id, ty::ClosureSubsts {
        substs: InternalSubsts::bound_vars_for_item(tcx, def_id),
    })
}

crate fn generator(tcx: TyCtxt<'tcx>, def_id: DefId) -> Ty<'tcx> {
    tcx.mk_generator(def_id, ty::GeneratorSubsts {
        substs: InternalSubsts::bound_vars_for_item(tcx, def_id),
    }, hir::GeneratorMovability::Movable)
}
