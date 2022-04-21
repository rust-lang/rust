use super::eval_queries::{mk_eval_cx, op_to_const};
use super::machine::CompileTimeEvalContext;
use crate::interpret::{
    intern_const_alloc_recursive, ConstValue, ImmTy, Immediate, InternKind, MemoryKind, PlaceTy,
    Scalar, ScalarMaybeUninit,
};
use rustc_middle::mir::interpret::ConstAlloc;
use rustc_middle::mir::{Field, ProjectionElem};
use rustc_middle::ty::{self, ScalarInt, Ty, TyCtxt};
use rustc_span::source_map::DUMMY_SP;
use rustc_target::abi::VariantIdx;

use crate::interpret::MPlaceTy;
use crate::interpret::Value;

/// Convert an evaluated constant to a type level constant
#[instrument(skip(tcx), level = "debug")]
pub(crate) fn const_to_valtree<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    raw: ConstAlloc<'tcx>,
) -> Option<ty::ValTree<'tcx>> {
    let ecx = mk_eval_cx(
        tcx, DUMMY_SP, param_env,
        // It is absolutely crucial for soundness that
        // we do not read from static items or other mutable memory.
        false,
    );
    let place = ecx.raw_const_to_mplace(raw).unwrap();
    const_to_valtree_inner(&ecx, &place)
}

#[instrument(skip(ecx), level = "debug")]
fn branches<'tcx>(
    ecx: &CompileTimeEvalContext<'tcx, 'tcx>,
    place: &MPlaceTy<'tcx>,
    n: usize,
    variant: Option<VariantIdx>,
) -> Option<ty::ValTree<'tcx>> {
    let place = match variant {
        Some(variant) => ecx.mplace_downcast(&place, variant).unwrap(),
        None => *place,
    };
    let variant = variant.map(|variant| Some(ty::ValTree::Leaf(ScalarInt::from(variant.as_u32()))));
    debug!(?place, ?variant);

    let fields = (0..n).map(|i| {
        let field = ecx.mplace_field(&place, i).unwrap();
        const_to_valtree_inner(ecx, &field)
    });
    // For enums, we preped their variant index before the variant's fields so we can figure out
    // the variant again when just seeing a valtree.
    let branches = variant.into_iter().chain(fields);
    Some(ty::ValTree::Branch(ecx.tcx.arena.alloc_from_iter(branches.collect::<Option<Vec<_>>>()?)))
}

#[instrument(skip(ecx), level = "debug")]
fn slice_branches<'tcx>(
    ecx: &CompileTimeEvalContext<'tcx, 'tcx>,
    place: &MPlaceTy<'tcx>,
) -> Option<ty::ValTree<'tcx>> {
    let n = place.len(&ecx.tcx.tcx).expect(&format!("expected to use len of place {:?}", place));
    let branches = (0..n).map(|i| {
        let place_elem = ecx.mplace_index(place, i).unwrap();
        const_to_valtree_inner(ecx, &place_elem)
    });

    Some(ty::ValTree::Branch(ecx.tcx.arena.alloc_from_iter(branches.collect::<Option<Vec<_>>>()?)))
}

#[instrument(skip(ecx), level = "debug")]
fn const_to_valtree_inner<'tcx>(
    ecx: &CompileTimeEvalContext<'tcx, 'tcx>,
    place: &MPlaceTy<'tcx>,
) -> Option<ty::ValTree<'tcx>> {
    match place.layout.ty.kind() {
        ty::FnDef(..) => Some(ty::ValTree::zst()),
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char => {
            let val = ecx.read_immediate(&place.into()).unwrap();
            let val = val.to_scalar().unwrap();
            Some(ty::ValTree::Leaf(val.assert_int()))
        }

        // Raw pointers are not allowed in type level constants, as we cannot properly test them for
        // equality at compile-time (see `ptr_guaranteed_eq`/`_ne`).
        // Technically we could allow function pointers (represented as `ty::Instance`), but this is not guaranteed to
        // agree with runtime equality tests.
        ty::FnPtr(_) | ty::RawPtr(_) => None,

        ty::Ref(_, _, _)  => {
            let derefd_place = ecx.deref_operand(&place.into()).unwrap_or_else(|e| bug!("couldn't deref {:?}, error: {:?}", place, e));
            debug!(?derefd_place);

            const_to_valtree_inner(ecx, &derefd_place)
        }

        ty::Str | ty::Slice(_) | ty::Array(_, _) => {
            let valtree = slice_branches(ecx, place);
            debug!(?valtree);

            valtree
        }
        // Trait objects are not allowed in type level constants, as we have no concept for
        // resolving their backing type, even if we can do that at const eval time. We may
        // hypothetically be able to allow `dyn StructuralEq` trait objects in the future,
        // but it is unclear if this is useful.
        ty::Dynamic(..) => None,

        ty::Tuple(substs) => branches(ecx, place, substs.len(), None),

        ty::Adt(def, _) => {
            if def.variants().is_empty() {
                bug!("uninhabited types should have errored and never gotten converted to valtree")
            }

            let variant = ecx.read_discriminant(&place.into()).unwrap().1;

            branches(ecx, place, def.variant(variant).fields.len(), def.is_enum().then_some(variant))
        }

        ty::Never
        | ty::Error(_)
        | ty::Foreign(..)
        | ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Projection(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        // FIXME(oli-obk): we could look behind opaque types
        | ty::Opaque(..)
        | ty::Infer(_)
        // FIXME(oli-obk): we can probably encode closures just like structs
        | ty::Closure(..)
        | ty::Generator(..)
        | ty::GeneratorWitness(..) => None,
    }
}

#[instrument(skip(ecx), level = "debug")]
fn create_mplace_from_layout<'tcx>(
    ecx: &mut CompileTimeEvalContext<'tcx, 'tcx>,
    ty: Ty<'tcx>,
) -> MPlaceTy<'tcx> {
    let tcx = ecx.tcx;
    let param_env = ecx.param_env;
    let layout = tcx.layout_of(param_env.and(ty)).unwrap();
    debug!(?layout);

    ecx.allocate(layout, MemoryKind::Stack).unwrap()
}

#[instrument(skip(ecx), level = "debug")]
fn create_pointee_place<'tcx>(
    ecx: &mut CompileTimeEvalContext<'tcx, 'tcx>,
    ty: Ty<'tcx>,
    valtree: ty::ValTree<'tcx>,
) -> MPlaceTy<'tcx> {
    let tcx = ecx.tcx.tcx;

    match ty.kind() {
        ty::Slice(_) | ty::Str => {
            let slice_ty = match ty.kind() {
                ty::Slice(slice_ty) => *slice_ty,
                ty::Str => tcx.mk_ty(ty::Uint(ty::UintTy::U8)),
                _ => bug!("expected ty::Slice | ty::Str"),
            };

            // Create a place for the underlying array
            let len = valtree.unwrap_branch().len() as u64;
            let arr_ty = tcx.mk_array(slice_ty, len as u64);
            let place = create_mplace_from_layout(ecx, arr_ty);
            debug!(?place);

            place
        }
        _ => create_mplace_from_layout(ecx, ty),
    }
}

/// Converts a `ValTree` to a `ConstValue`, which is needed after mir
/// construction has finished.
#[instrument(skip(tcx), level = "debug")]
pub fn valtree_to_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env_ty: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
    valtree: ty::ValTree<'tcx>,
) -> ConstValue<'tcx> {
    // Basic idea: We directly construct `Scalar` values from trivial `ValTree`s
    // (those for constants with type bool, int, uint, float or char).
    // For all other types we create an `MPlace` and fill that by walking
    // the `ValTree` and using `place_projection` and `place_field` to
    // create inner `MPlace`s which are filled recursively.
    // FIXME Does this need an example?

    let (param_env, ty) = param_env_ty.into_parts();
    let mut ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);

    match ty.kind() {
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char => match valtree {
            ty::ValTree::Leaf(scalar_int) => ConstValue::Scalar(Scalar::Int(scalar_int)),
            ty::ValTree::Branch(_) => bug!(
                "ValTrees for Bool, Int, Uint, Float or Char should have the form ValTree::Leaf"
            ),
        },
        ty::Ref(_, inner_ty, _) => {
            // create a place for the pointee
            let mut pointee_place = create_pointee_place(&mut ecx, *inner_ty, valtree);
            debug!(?pointee_place);

            // insert elements of valtree into `place`
            fill_place_recursively(&mut ecx, &mut pointee_place, valtree);
            dump_place(&ecx, pointee_place.into());
            intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &pointee_place).unwrap();

            let ref_place = pointee_place.to_ref(&tcx);
            let imm = ImmTy::from_immediate(ref_place, tcx.layout_of(param_env_ty).unwrap());

            let const_val = op_to_const(&ecx, &imm.into());
            debug!(?const_val);

            const_val
        }
        ty::Tuple(_) | ty::Array(_, _) | ty::Adt(..) => {
            let mut place = create_mplace_from_layout(&mut ecx, ty);
            debug!(?place);

            fill_place_recursively(&mut ecx, &mut place, valtree);
            dump_place(&ecx, place.into());
            intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &place).unwrap();

            let const_val = op_to_const(&ecx, &place.into());
            debug!(?const_val);

            const_val
        }
        ty::Never
        | ty::FnDef(..)
        | ty::Error(_)
        | ty::Foreign(..)
        | ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Projection(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        | ty::Opaque(..)
        | ty::Infer(_)
        | ty::Closure(..)
        | ty::Generator(..)
        | ty::GeneratorWitness(..)
        | ty::FnPtr(_)
        | ty::RawPtr(_)
        | ty::Str
        | ty::Slice(_)
        | ty::Dynamic(..) => bug!("no ValTree should have been created for type {:?}", ty.kind()),
    }
}

// FIXME Needs a better/correct name
#[instrument(skip(ecx), level = "debug")]
fn fill_place_recursively<'tcx>(
    ecx: &mut CompileTimeEvalContext<'tcx, 'tcx>,
    place: &mut MPlaceTy<'tcx>,
    valtree: ty::ValTree<'tcx>,
) {
    // This will match on valtree and write the value(s) corresponding to the ValTree
    // inside the place recursively.

    let tcx = ecx.tcx.tcx;
    let ty = place.layout.ty;

    match ty.kind() {
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char => {
            let scalar_int = valtree.unwrap_leaf();
            debug!("writing trivial valtree {:?} to place {:?}", scalar_int, place);
            ecx.write_immediate(
                Immediate::Scalar(ScalarMaybeUninit::Scalar(scalar_int.into())),
                &(*place).into(),
            )
            .unwrap();
        }
        ty::Ref(_, inner_ty, _) => {
            let mut pointee_place = create_pointee_place(ecx, *inner_ty, valtree);
            debug!(?pointee_place);

            fill_place_recursively(ecx, &mut pointee_place, valtree);
            dump_place(ecx, pointee_place.into());
            intern_const_alloc_recursive(ecx, InternKind::Constant, &pointee_place).unwrap();

            let imm = match inner_ty.kind() {
                ty::Slice(_) | ty::Str => {
                    let len = valtree.unwrap_branch().len();
                    let len_scalar = ScalarMaybeUninit::Scalar(Scalar::from_u64(len as u64));

                    Immediate::ScalarPair(
                        ScalarMaybeUninit::from_maybe_pointer((*pointee_place).ptr, &tcx),
                        len_scalar,
                    )
                }
                _ => pointee_place.to_ref(&tcx),
            };
            debug!(?imm);

            ecx.write_immediate(imm, &(*place).into()).unwrap();
        }
        ty::Adt(_, _) | ty::Tuple(_) | ty::Array(_, _) | ty::Str => {
            let branches = valtree.unwrap_branch();

            // Need to downcast place for enums
            let (place_adjusted, branches, variant_idx) = match ty.kind() {
                ty::Adt(def, _) if def.is_enum() => {
                    // First element of valtree corresponds to variant
                    let scalar_int = branches[0].unwrap_leaf();
                    let variant_idx = VariantIdx::from_u32(scalar_int.try_to_u32().unwrap());
                    let variant = def.variant(variant_idx);
                    debug!(?variant);

                    (
                        place.project_downcast(ecx, variant_idx).unwrap(),
                        &branches[1..],
                        Some(variant_idx),
                    )
                }
                _ => (*place, branches, None),
            };
            debug!(?place_adjusted, ?branches);

            // Create the places for the fields and fill them recursively
            for (i, inner_valtree) in branches.iter().enumerate() {
                debug!(?i, ?inner_valtree);

                let mut place_inner = match *ty.kind() {
                    ty::Adt(def, substs) if !def.is_enum() => {
                        let field = &def.variant(VariantIdx::from_usize(0)).fields[i];
                        let field_ty = field.ty(tcx, substs);
                        let projection_elem = ProjectionElem::Field(Field::from_usize(i), field_ty);

                        ecx.mplace_projection(&place_adjusted, projection_elem).unwrap()
                    }
                    ty::Adt(_, _) | ty::Tuple(_) => ecx.mplace_field(&place_adjusted, i).unwrap(),
                    ty::Array(_, _) | ty::Str => {
                        ecx.mplace_index(&place_adjusted, i as u64).unwrap()
                    }
                    _ => bug!(),
                };
                debug!(?place_inner);

                // insert valtree corresponding to tuple element into place
                fill_place_recursively(ecx, &mut place_inner, *inner_valtree);
                dump_place(&ecx, place_inner.into());
            }

            debug!("dump of place_adjusted:");
            dump_place(ecx, place_adjusted.into());

            if let Some(variant_idx) = variant_idx {
                // don't forget filling the place with the discriminant of the enum
                ecx.write_discriminant(variant_idx, &(*place).into()).unwrap();
            }

            dump_place(ecx, (*place).into());
        }
        _ => bug!("shouldn't have created a ValTree for {:?}", ty),
    }
}

fn dump_place<'tcx>(ecx: &CompileTimeEvalContext<'tcx, 'tcx>, place: PlaceTy<'tcx>) {
    trace!("{:?}", ecx.dump_place(*place));
}
