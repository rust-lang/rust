use rustc_hir::def::DefKind;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Representability, Ty, TyCtxt};
use rustc_span::def_id::LocalDefId;

pub(crate) fn provide(providers: &mut Providers) {
    *providers =
        Providers { representability, representability_adt_ty, params_in_repr, ..*providers };
}

macro_rules! rtry {
    ($e:expr) => {
        match $e {
            e @ Representability::Infinite(_) => return e,
            Representability::Representable => {}
        }
    };
}

fn representability(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Representability {
    match tcx.def_kind(def_id) {
        DefKind::Struct | DefKind::Union | DefKind::Enum => {
            for variant in tcx.adt_def(def_id).variants() {
                for field in variant.fields.iter() {
                    rtry!(tcx.representability(field.did.expect_local()));
                }
            }
            Representability::Representable
        }
        DefKind::Field => representability_ty(tcx, tcx.type_of(def_id).instantiate_identity()),
        def_kind => bug!("unexpected {def_kind:?}"),
    }
}

fn representability_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Representability {
    match *ty.kind() {
        ty::Adt(..) => tcx.representability_adt_ty(ty),
        // FIXME(#11924) allow zero-length arrays?
        ty::Array(ty, _) => representability_ty(tcx, ty),
        ty::Tuple(tys) => {
            for ty in tys {
                rtry!(representability_ty(tcx, ty));
            }
            Representability::Representable
        }
        _ => Representability::Representable,
    }
}

/*
The reason for this being a separate query is very subtle:
Consider this infinitely sized struct: `struct Foo(Box<Foo>, Bar<Foo>)`:
When calling representability(Foo), a query cycle will occur:
  representability(Foo)
    -> representability_adt_ty(Bar<Foo>)
    -> representability(Foo)
For the diagnostic output (in `Value::from_cycle_error`), we want to detect that
the `Foo` in the *second* field of the struct is culpable. This requires
traversing the HIR of the struct and calling `params_in_repr(Bar)`. But we can't
call params_in_repr for a given type unless it is known to be representable.
params_in_repr will cycle/panic on infinitely sized types. Looking at the query
cycle above, we know that `Bar` is representable because
representability_adt_ty(Bar<..>) is in the cycle and representability(Bar) is
*not* in the cycle.
*/
fn representability_adt_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Representability {
    let ty::Adt(adt, args) = ty.kind() else { bug!("expected adt") };
    if let Some(def_id) = adt.did().as_local() {
        rtry!(tcx.representability(def_id));
    }
    // At this point, we know that the item of the ADT type is representable;
    // but the type parameters may cause a cycle with an upstream type
    let params_in_repr = tcx.params_in_repr(adt.did());
    for (i, arg) in args.iter().enumerate() {
        if let ty::GenericArgKind::Type(ty) = arg.kind() {
            if params_in_repr.contains(i as u32) {
                rtry!(representability_ty(tcx, ty));
            }
        }
    }
    Representability::Representable
}

fn params_in_repr(tcx: TyCtxt<'_>, def_id: LocalDefId) -> DenseBitSet<u32> {
    let adt_def = tcx.adt_def(def_id);
    let generics = tcx.generics_of(def_id);
    let mut params_in_repr = DenseBitSet::new_empty(generics.own_params.len());
    for variant in adt_def.variants() {
        for field in variant.fields.iter() {
            params_in_repr_ty(
                tcx,
                tcx.type_of(field.did).instantiate_identity(),
                &mut params_in_repr,
            );
        }
    }
    params_in_repr
}

fn params_in_repr_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, params_in_repr: &mut DenseBitSet<u32>) {
    match *ty.kind() {
        ty::Adt(adt, args) => {
            let inner_params_in_repr = tcx.params_in_repr(adt.did());
            for (i, arg) in args.iter().enumerate() {
                if let ty::GenericArgKind::Type(ty) = arg.kind() {
                    if inner_params_in_repr.contains(i as u32) {
                        params_in_repr_ty(tcx, ty, params_in_repr);
                    }
                }
            }
        }
        ty::Array(ty, _) => params_in_repr_ty(tcx, ty, params_in_repr),
        ty::Tuple(tys) => tys.iter().for_each(|ty| params_in_repr_ty(tcx, ty, params_in_repr)),
        ty::Param(param) => {
            params_in_repr.insert(param.index);
        }
        _ => {}
    }
}
