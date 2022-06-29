use rustc_middle::ty::{self, TyCtxt};
use rustc_target::abi::VariantIdx;

use std::iter;

/// Destructures array, ADT or tuple constants into the constants
/// of their fields.
pub(crate) fn destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    const_: ty::Const<'tcx>,
) -> ty::DestructuredConst<'tcx> {
    let ty::ConstKind::Value(valtree) = const_.kind() else {
        bug!("cannot destructure constant {:?}", const_)
    };

    let branches = match valtree {
        ty::ValTree::Branch(b) => b,
        _ => bug!("cannot destructure constant {:?}", const_),
    };

    let (fields, variant) = match const_.ty().kind() {
        ty::Array(inner_ty, _) | ty::Slice(inner_ty) => {
            // construct the consts for the elements of the array/slice
            let field_consts = branches
                .iter()
                .map(|b| tcx.mk_const(ty::ConstS { kind: ty::ConstKind::Value(*b), ty: *inner_ty }))
                .collect::<Vec<_>>();
            debug!(?field_consts);

            (field_consts, None)
        }
        ty::Adt(def, _) if def.variants().is_empty() => bug!("unreachable"),
        ty::Adt(def, substs) => {
            let (variant_idx, branches) = if def.is_enum() {
                let (head, rest) = branches.split_first().unwrap();
                (VariantIdx::from_u32(head.unwrap_leaf().try_to_u32().unwrap()), rest)
            } else {
                (VariantIdx::from_u32(0), branches)
            };
            let fields = &def.variant(variant_idx).fields;
            let mut field_consts = Vec::with_capacity(fields.len());

            for (field, field_valtree) in iter::zip(fields, branches) {
                let field_ty = field.ty(tcx, substs);
                let field_const = tcx.mk_const(ty::ConstS {
                    kind: ty::ConstKind::Value(*field_valtree),
                    ty: field_ty,
                });
                field_consts.push(field_const);
            }
            debug!(?field_consts);

            (field_consts, Some(variant_idx))
        }
        ty::Tuple(elem_tys) => {
            let fields = iter::zip(*elem_tys, branches)
                .map(|(elem_ty, elem_valtree)| {
                    tcx.mk_const(ty::ConstS {
                        kind: ty::ConstKind::Value(*elem_valtree),
                        ty: elem_ty,
                    })
                })
                .collect::<Vec<_>>();

            (fields, None)
        }
        _ => bug!("cannot destructure constant {:?}", const_),
    };

    let fields = tcx.arena.alloc_from_iter(fields.into_iter());

    ty::DestructuredConst { variant, fields }
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { destructure_const, ..*providers };
}
