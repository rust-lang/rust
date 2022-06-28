use rustc_middle::ty::{self, TyCtxt};
use rustc_target::abi::VariantIdx;

/// Tries to destructure constants of type Array or Adt into the constants
/// of its fields.
pub(crate) fn destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    const_: ty::Const<'tcx>,
) -> ty::DestructuredConst<'tcx> {
    if let ty::ConstKind::Value(valtree) = const_.kind() {
        let branches = match valtree {
            ty::ValTree::Branch(b) => b,
            _ => bug!("cannot destructure constant {:?}", const_),
        };

        let (fields, variant) = match const_.ty().kind() {
            ty::Array(inner_ty, _) | ty::Slice(inner_ty) => {
                // construct the consts for the elements of the array/slice
                let field_consts = branches
                    .iter()
                    .map(|b| {
                        tcx.mk_const(ty::ConstS { kind: ty::ConstKind::Value(*b), ty: *inner_ty })
                    })
                    .collect::<Vec<_>>();
                debug!(?field_consts);

                (field_consts, None)
            }
            ty::Adt(def, _) if def.variants().is_empty() => bug!("unreachable"),
            ty::Adt(def, substs) => {
                let variant_idx = if def.is_enum() {
                    VariantIdx::from_u32(branches[0].unwrap_leaf().try_to_u32().unwrap())
                } else {
                    VariantIdx::from_u32(0)
                };
                let fields = &def.variant(variant_idx).fields;
                let mut field_consts = Vec::with_capacity(fields.len());

                // Note: First element inValTree corresponds to variant of enum
                let mut valtree_idx = if def.is_enum() { 1 } else { 0 };
                for field in fields {
                    let field_ty = field.ty(tcx, substs);
                    let field_valtree = branches[valtree_idx]; // first element of branches is variant
                    let field_const = tcx.mk_const(ty::ConstS {
                        kind: ty::ConstKind::Value(field_valtree),
                        ty: field_ty,
                    });
                    field_consts.push(field_const);
                    valtree_idx += 1;
                }
                debug!(?field_consts);

                (field_consts, Some(variant_idx))
            }
            ty::Tuple(elem_tys) => {
                let fields = elem_tys
                    .iter()
                    .enumerate()
                    .map(|(i, elem_ty)| {
                        let elem_valtree = branches[i];
                        tcx.mk_const(ty::ConstS {
                            kind: ty::ConstKind::Value(elem_valtree),
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
    } else {
        bug!("cannot destructure constant {:?}", const_)
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers =
        ty::query::Providers { destructure_const, ..*providers };
}
