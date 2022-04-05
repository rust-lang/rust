use super::eval_queries::{mk_eval_cx, op_to_const};
use super::machine::CompileTimeEvalContext;
use crate::interpret::{
    intern_const_alloc_recursive, ConstValue, ImmTy, Immediate, InternKind, MemoryKind, PlaceTy,
    Pointer, Scalar, ScalarMaybeUninit,
};
use rustc_middle::mir::interpret::{ConstAlloc, GlobalAlloc};
use rustc_middle::mir::{Field, ProjectionElem};
use rustc_middle::ty::{self, ScalarInt, Ty, TyCtxt};
use rustc_span::source_map::DUMMY_SP;
use rustc_target::abi::VariantIdx;

use crate::interpret::visitor::Value;
use crate::interpret::MPlaceTy;

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
    param_env_ty: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> MPlaceTy<'tcx> {
    let tcx = ecx.tcx;
    let layout = tcx.layout_of(param_env_ty).unwrap();
    debug!(?layout);

    ecx.allocate(layout, MemoryKind::Stack).unwrap()
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
            match inner_ty.kind() {
                ty::Slice(_) | ty::Str => {
                    let slice_ty = match inner_ty.kind() {
                        ty::Slice(slice_ty) => *slice_ty,
                        ty::Str => tcx.mk_ty(ty::Uint(ty::UintTy::U8)),
                        _ => bug!("expected ty::Slice | ty::Str"),
                    };
                    debug!(?slice_ty);

                    let valtrees = valtree.unwrap_branch();

                    // Create a place for the underlying array
                    let len = valtrees.len();
                    let arr_ty = tcx.mk_array(slice_ty, len as u64);
                    let mut place =
                        create_mplace_from_layout(&mut ecx, ty::ParamEnv::empty().and(arr_ty));
                    debug!(?place);

                    // Insert elements of `arr_valtree` into `place`
                    fill_place_recursively(&mut ecx, &mut place, valtree, arr_ty);
                    dump_place(&ecx, place.into());

                    // The allocation behind `place` is local, we need to intern it
                    intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &place).unwrap();

                    // Now we need to get the Allocation
                    let alloc_id = place.mplace.ptr.provenance.unwrap();
                    debug!(?alloc_id);

                    let data = match tcx.get_global_alloc(alloc_id) {
                        Some(GlobalAlloc::Memory(const_alloc)) => const_alloc,
                        _ => bug!("expected memory allocation"),
                    };
                    debug!(?data);

                    return ConstValue::Slice { data, start: 0, end: len as usize };
                }
                _ => {
                    match valtree {
                        ty::ValTree::Branch(_) => {
                            // create a place for the pointee
                            let mut place = create_mplace_from_layout(
                                &mut ecx,
                                ty::ParamEnv::empty().and(*inner_ty),
                            );
                            debug!(?place);

                            // insert elements of valtree into `place`
                            fill_place_recursively(&mut ecx, &mut place, valtree, *inner_ty);
                            dump_place(&ecx, place.into());
                            intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &place)
                                .unwrap();

                            let ref_place = place.mplace.to_ref(&tcx);
                            let imm = ImmTy::from_immediate(
                                ref_place,
                                tcx.layout_of(param_env_ty).unwrap(),
                            );

                            let const_val = op_to_const(&ecx, &imm.into());
                            debug!(?const_val);

                            const_val
                        }
                        ty::ValTree::Leaf(_) => {
                            let mut place = create_mplace_from_layout(
                                &mut ecx,
                                ty::ParamEnv::empty().and(*inner_ty),
                            );

                            fill_place_recursively(&mut ecx, &mut place, valtree, *inner_ty);
                            dump_place(&ecx, place.into());

                            let ref_place = place.mplace.to_ref(&tcx);
                            let imm = ImmTy::from_immediate(
                                ref_place,
                                tcx.layout_of(param_env_ty).unwrap(),
                            );

                            op_to_const(&ecx, &imm.into())
                        }
                    }
                }
            }
        }
        ty::Tuple(_) | ty::Array(_, _) | ty::Adt(..) => {
            let mut place = create_mplace_from_layout(&mut ecx, param_env_ty);
            debug!(?place);

            fill_place_recursively(&mut ecx, &mut place, valtree, ty);
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
    ty: Ty<'tcx>,
) {
    // This will match on valtree and write the value(s) corresponding to the ValTree
    // inside the place recursively.

    let tcx = ecx.tcx.tcx;

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
            match inner_ty.kind() {
                ty::Slice(_) | ty::Str => {
                    let slice_ty = match inner_ty.kind() {
                        ty::Slice(slice_ty) => *slice_ty,
                        ty::Str => tcx.mk_ty(ty::Uint(ty::UintTy::U8)),
                        _ => bug!("expected ty::Slice | ty::Str"),
                    };
                    debug!(?slice_ty);

                    let valtrees = valtree.unwrap_branch();
                    debug!(?valtrees);
                    let len = valtrees.len();
                    debug!(?len);

                    // create a place for the underlying array
                    let arr_ty = tcx.mk_array(slice_ty, len as u64);
                    let mut arr_place =
                        create_mplace_from_layout(ecx, ty::ParamEnv::empty().and(arr_ty));
                    debug!(?arr_place);

                    // Insert elements of `arr_valtree` into `place`
                    fill_place_recursively(ecx, &mut arr_place, valtree, arr_ty);
                    dump_place(&ecx, arr_place.into());

                    // Now we need to create a `ScalarPair` from the filled `place`
                    // and write that into `place`
                    let (alloc_id, offset) = arr_place.mplace.ptr.into_parts();
                    debug!(?alloc_id, ?offset);
                    let unwrapped_ptr = Pointer { offset, provenance: alloc_id.unwrap() };
                    let len_scalar = ScalarMaybeUninit::Scalar(Scalar::from_u64(len as u64));

                    let imm = Immediate::ScalarPair(
                        ScalarMaybeUninit::from_pointer(unwrapped_ptr, &tcx),
                        len_scalar,
                    );
                    debug!(?imm);

                    // Now write the ScalarPair into the original place we wanted to fill
                    // in this call
                    let _ = ecx.write_immediate(imm, &(*place).into()).unwrap();

                    dump_place(&ecx, (*place).into());
                }
                _ => {
                    let mut pointee_place =
                        create_mplace_from_layout(ecx, ty::ParamEnv::empty().and(*inner_ty));
                    debug!(?pointee_place);
                    fill_place_recursively(ecx, &mut pointee_place, valtree, *inner_ty);

                    dump_place(ecx, pointee_place.into());
                    intern_const_alloc_recursive(ecx, InternKind::Constant, &pointee_place)
                        .unwrap();

                    let imm = pointee_place.mplace.to_ref(&tcx);
                    debug!(?imm);

                    ecx.write_immediate(imm, &(*place).into()).unwrap();
                }
            }
        }
        ty::Tuple(tuple_types) => {
            let branches = valtree.unwrap_branch();
            assert_eq!(tuple_types.len(), branches.len());

            for (i, inner_valtree) in branches.iter().enumerate() {
                debug!(?i, ?inner_valtree);
                let inner_ty = tuple_types.get(i).expect(&format!(
                    "expected to be able to index at position {} into {:?}",
                    i, tuple_types
                ));
                debug!(?inner_ty);

                // Create the mplace for the tuple element
                let mut place_inner = ecx.mplace_field(place, i).unwrap();
                debug!(?place_inner);

                // insert valtree corresponding to tuple element into place
                fill_place_recursively(ecx, &mut place_inner, *inner_valtree, *inner_ty);
            }
        }
        ty::Array(inner_ty, _) => {
            let inner_valtrees = valtree.unwrap_branch();
            for (i, inner_valtree) in inner_valtrees.iter().enumerate() {
                debug!(?i, ?inner_valtree);

                let mut place_inner = ecx.mplace_field(place, i).unwrap();
                debug!(?place_inner);

                fill_place_recursively(ecx, &mut place_inner, *inner_valtree, *inner_ty)
            }
        }
        ty::Adt(def, substs) if def.is_enum() => {
            debug!("enum, substs: {:?}", substs);
            let inner_valtrees = valtree.unwrap_branch();

            // First element of valtree corresponds to variant
            let scalar_int = inner_valtrees[0].unwrap_leaf();
            let variant_idx = VariantIdx::from_u32(scalar_int.try_to_u32().unwrap());
            let variant = def.variant(variant_idx);
            debug!(?variant);

            // Need to downcast place
            let place_downcast = place.project_downcast(ecx, variant_idx).unwrap();
            debug!(?place_downcast);

            // fill `place_downcast` with the valtree elements corresponding to
            // the fields of the enum
            let fields = &variant.fields;
            let inner_valtrees = &inner_valtrees[1..];
            for (i, field) in fields.iter().enumerate() {
                debug!(?i, ?field);

                let field_ty = field.ty(tcx, substs);
                debug!(?field_ty);

                let mut field_mplace = ecx.mplace_field(&place_downcast, i).unwrap();
                debug!(?field_mplace);
                let inner_valtree = inner_valtrees[i];

                fill_place_recursively(ecx, &mut field_mplace, inner_valtree, field_ty);
                dump_place(&ecx, field_mplace.into());
            }

            debug!("dump of place_downcast");
            dump_place(ecx, place_downcast.into());

            // don't forget filling the place with the discriminant of the enum
            ecx.write_discriminant(variant_idx, &(*place).into()).unwrap();
            dump_place(ecx, (*place).into());
        }
        ty::Adt(def, substs) => {
            debug!("Adt def: {:?} with substs: {:?}", def, substs);
            let inner_valtrees = valtree.unwrap_branch();
            debug!(?inner_valtrees);
            let (fields, inner_valtrees) =
                (&def.variant(VariantIdx::from_usize(0)).fields[..], inner_valtrees);

            debug!("fields: {:?}", fields);

            for (i, field) in fields.iter().enumerate() {
                let field_ty = field.ty(tcx, substs);
                debug!(?field_ty);
                let old_field_ty = tcx.type_of(field.did);
                debug!(?old_field_ty);
                let projection_elem = ProjectionElem::Field(Field::from_usize(i), field_ty);
                let mut field_place = ecx.mplace_projection(place, projection_elem).unwrap();
                let inner_valtree = inner_valtrees[i];

                fill_place_recursively(ecx, &mut field_place, inner_valtree, field_ty);
            }
        }
        _ => {}
    }
}

fn dump_place<'tcx>(ecx: &CompileTimeEvalContext<'tcx, 'tcx>, place: PlaceTy<'tcx>) {
    trace!("{:?}", ecx.dump_place(place.place));
}
