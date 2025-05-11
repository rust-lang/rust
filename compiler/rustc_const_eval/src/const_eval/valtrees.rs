use rustc_abi::{BackendRepr, VariantIdx};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_middle::mir::interpret::{EvalToValTreeResult, GlobalId, ReportedErrorInfo};
use rustc_middle::ty::layout::{LayoutCx, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_span::DUMMY_SP;
use tracing::{debug, instrument, trace};

use super::eval_queries::{mk_eval_cx_to_read_const_val, op_to_const};
use super::machine::CompileTimeInterpCx;
use super::{VALTREE_MAX_NODES, ValTreeCreationError, ValTreeCreationResult};
use crate::const_eval::CanAccessMutGlobal;
use crate::errors::MaxNumNodesInConstErr;
use crate::interpret::{
    ImmTy, Immediate, InternKind, MPlaceTy, MemPlaceMeta, MemoryKind, PlaceTy, Projectable, Scalar,
    intern_const_alloc_recursive,
};

#[instrument(skip(ecx), level = "debug")]
fn branches<'tcx>(
    ecx: &CompileTimeInterpCx<'tcx>,
    place: &MPlaceTy<'tcx>,
    field_count: usize,
    variant: Option<VariantIdx>,
    num_nodes: &mut usize,
) -> ValTreeCreationResult<'tcx> {
    let place = match variant {
        Some(variant) => ecx.project_downcast(place, variant).unwrap(),
        None => place.clone(),
    };
    debug!(?place);

    let mut branches = Vec::with_capacity(field_count + variant.is_some() as usize);

    // For enums, we prepend their variant index before the variant's fields so we can figure out
    // the variant again when just seeing a valtree.
    if let Some(variant) = variant {
        branches.push(ty::ValTree::from_scalar_int(*ecx.tcx, variant.as_u32().into()));
    }

    for i in 0..field_count {
        let field = ecx.project_field(&place, i).unwrap();
        let valtree = const_to_valtree_inner(ecx, &field, num_nodes)?;
        branches.push(valtree);
    }

    // Have to account for ZSTs here
    if branches.len() == 0 {
        *num_nodes += 1;
    }

    Ok(ty::ValTree::from_branches(*ecx.tcx, branches))
}

#[instrument(skip(ecx), level = "debug")]
fn slice_branches<'tcx>(
    ecx: &CompileTimeInterpCx<'tcx>,
    place: &MPlaceTy<'tcx>,
    num_nodes: &mut usize,
) -> ValTreeCreationResult<'tcx> {
    let n = place.len(ecx).unwrap_or_else(|_| panic!("expected to use len of place {place:?}"));

    let mut elems = Vec::with_capacity(n as usize);
    for i in 0..n {
        let place_elem = ecx.project_index(place, i).unwrap();
        let valtree = const_to_valtree_inner(ecx, &place_elem, num_nodes)?;
        elems.push(valtree);
    }

    Ok(ty::ValTree::from_branches(*ecx.tcx, elems))
}

#[instrument(skip(ecx), level = "debug")]
fn const_to_valtree_inner<'tcx>(
    ecx: &CompileTimeInterpCx<'tcx>,
    place: &MPlaceTy<'tcx>,
    num_nodes: &mut usize,
) -> ValTreeCreationResult<'tcx> {
    let tcx = *ecx.tcx;
    let ty = place.layout.ty;
    debug!("ty kind: {:?}", ty.kind());

    if *num_nodes >= VALTREE_MAX_NODES {
        return Err(ValTreeCreationError::NodesOverflow);
    }

    match ty.kind() {
        ty::FnDef(..) => {
            *num_nodes += 1;
            Ok(ty::ValTree::zst(tcx))
        }
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char => {
            let val = ecx.read_immediate(place).unwrap();
            let val = val.to_scalar_int().unwrap();
            *num_nodes += 1;

            Ok(ty::ValTree::from_scalar_int(tcx, val))
        }

        ty::Pat(base, ..) => {
            let mut place = place.clone();
            // The valtree of the base type is the same as the valtree of the pattern type.
            // Since the returned valtree does not contain the type or layout, we can just
            // switch to the base type.
            place.layout = ecx.layout_of(*base).unwrap();
            ensure_sufficient_stack(|| const_to_valtree_inner(ecx, &place, num_nodes))
        },


        ty::RawPtr(_, _) => {
            // Not all raw pointers are allowed, as we cannot properly test them for
            // equality at compile-time (see `ptr_guaranteed_cmp`).
            // However we allow those that are just integers in disguise.
            // First, get the pointer. Remember it might be wide!
            let val = ecx.read_immediate(place).unwrap();
            // We could allow wide raw pointers where both sides are integers in the future,
            // but for now we reject them.
            if matches!(val.layout.backend_repr, BackendRepr::ScalarPair(..)) {
                return Err(ValTreeCreationError::NonSupportedType(ty));
            }
            let val = val.to_scalar();
            // We are in the CTFE machine, so ptr-to-int casts will fail.
            // This can only be `Ok` if `val` already is an integer.
            let Ok(val) = val.try_to_scalar_int() else {
                return Err(ValTreeCreationError::NonSupportedType(ty));
            };
            // It's just a ScalarInt!
            Ok(ty::ValTree::from_scalar_int(tcx, val))
        }

        // Technically we could allow function pointers (represented as `ty::Instance`), but this is not guaranteed to
        // agree with runtime equality tests.
        ty::FnPtr(..) => Err(ValTreeCreationError::NonSupportedType(ty)),

        ty::Ref(_, _, _)  => {
            let derefd_place = ecx.deref_pointer(place).unwrap();
            const_to_valtree_inner(ecx, &derefd_place, num_nodes)
        }

        ty::Str | ty::Slice(_) | ty::Array(_, _) => {
            slice_branches(ecx, place, num_nodes)
        }
        // Trait objects are not allowed in type level constants, as we have no concept for
        // resolving their backing type, even if we can do that at const eval time. We may
        // hypothetically be able to allow `dyn StructuralPartialEq` trait objects in the future,
        // but it is unclear if this is useful.
        ty::Dynamic(..) => Err(ValTreeCreationError::NonSupportedType(ty)),

        ty::Tuple(elem_tys) => {
            branches(ecx, place, elem_tys.len(), None, num_nodes)
        }

        ty::Adt(def, _) => {
            if def.is_union() {
                return Err(ValTreeCreationError::NonSupportedType(ty));
            } else if def.variants().is_empty() {
                bug!("uninhabited types should have errored and never gotten converted to valtree")
            }

            let variant = ecx.read_discriminant(place).unwrap();
            branches(ecx, place, def.variant(variant).fields.len(), def.is_enum().then_some(variant), num_nodes)
        }

        ty::Never
        | ty::Error(_)
        | ty::Foreign(..)
        | ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        // FIXME(oli-obk): we could look behind opaque types
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        | ty::Infer(_)
        // FIXME(oli-obk): we can probably encode closures just like structs
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Coroutine(..)
        | ty::CoroutineWitness(..)
        | ty::UnsafeBinder(_) => Err(ValTreeCreationError::NonSupportedType(ty)),
    }
}

/// Valtrees don't store the `MemPlaceMeta` that all dynamically sized values have in the interpreter.
/// This function reconstructs it.
fn reconstruct_place_meta<'tcx>(
    layout: TyAndLayout<'tcx>,
    valtree: ty::ValTree<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> MemPlaceMeta {
    if layout.is_sized() {
        return MemPlaceMeta::None;
    }

    let mut last_valtree = valtree;
    // Traverse the type, and update `last_valtree` as we go.
    let tail = tcx.struct_tail_raw(
        layout.ty,
        |ty| ty,
        || {
            let branches = last_valtree.unwrap_branch();
            last_valtree = *branches.last().unwrap();
            debug!(?branches, ?last_valtree);
        },
    );
    // Sanity-check that we got a tail we support.
    match tail.kind() {
        ty::Slice(..) | ty::Str => {}
        _ => bug!("unsized tail of a valtree must be Slice or Str"),
    };

    // Get the number of elements in the unsized field.
    let num_elems = last_valtree.unwrap_branch().len();
    MemPlaceMeta::Meta(Scalar::from_target_usize(num_elems as u64, &tcx))
}

#[instrument(skip(ecx), level = "debug", ret)]
fn create_valtree_place<'tcx>(
    ecx: &mut CompileTimeInterpCx<'tcx>,
    layout: TyAndLayout<'tcx>,
    valtree: ty::ValTree<'tcx>,
) -> MPlaceTy<'tcx> {
    let meta = reconstruct_place_meta(layout, valtree, ecx.tcx.tcx);
    ecx.allocate_dyn(layout, MemoryKind::Stack, meta).unwrap()
}

/// Evaluates a constant and turns it into a type-level constant value.
pub(crate) fn eval_to_valtree<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    cid: GlobalId<'tcx>,
) -> EvalToValTreeResult<'tcx> {
    // Const eval always happens in PostAnalysis mode . See the comment in
    // `InterpCx::new` for more details.
    debug_assert_eq!(typing_env.typing_mode, ty::TypingMode::PostAnalysis);
    let const_alloc = tcx.eval_to_allocation_raw(typing_env.as_query_input(cid))?;

    // FIXME Need to provide a span to `eval_to_valtree`
    let ecx = mk_eval_cx_to_read_const_val(
        tcx,
        DUMMY_SP,
        typing_env,
        // It is absolutely crucial for soundness that
        // we do not read from mutable memory.
        CanAccessMutGlobal::No,
    );
    let place = ecx.raw_const_to_mplace(const_alloc).unwrap();
    debug!(?place);

    let mut num_nodes = 0;
    let valtree_result = const_to_valtree_inner(&ecx, &place, &mut num_nodes);

    match valtree_result {
        Ok(valtree) => Ok(Ok(valtree)),
        Err(err) => {
            let did = cid.instance.def_id();
            let global_const_id = cid.display(tcx);
            let span = tcx.hir_span_if_local(did);
            match err {
                ValTreeCreationError::NodesOverflow => {
                    let handled =
                        tcx.dcx().emit_err(MaxNumNodesInConstErr { span, global_const_id });
                    Err(ReportedErrorInfo::allowed_in_infallible(handled).into())
                }
                ValTreeCreationError::NonSupportedType(ty) => Ok(Err(ty)),
            }
        }
    }
}

/// Converts a `ValTree` to a `ConstValue`, which is needed after mir
/// construction has finished.
// FIXME(valtrees): Merge `valtree_to_const_value` and `valtree_into_mplace` into one function
#[instrument(skip(tcx), level = "debug", ret)]
pub fn valtree_to_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    cv: ty::Value<'tcx>,
) -> mir::ConstValue<'tcx> {
    // Basic idea: We directly construct `Scalar` values from trivial `ValTree`s
    // (those for constants with type bool, int, uint, float or char).
    // For all other types we create an `MPlace` and fill that by walking
    // the `ValTree` and using `place_projection` and `place_field` to
    // create inner `MPlace`s which are filled recursively.
    // FIXME: Does this need an example?
    match *cv.ty.kind() {
        ty::FnDef(..) => {
            assert!(cv.valtree.is_zst());
            mir::ConstValue::ZeroSized
        }
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char | ty::RawPtr(_, _) => {
            mir::ConstValue::Scalar(Scalar::Int(cv.valtree.unwrap_leaf()))
        }
        ty::Pat(ty, _) => {
            let cv = ty::Value { valtree: cv.valtree, ty };
            valtree_to_const_value(tcx, typing_env, cv)
        }
        ty::Ref(_, inner_ty, _) => {
            let mut ecx =
                mk_eval_cx_to_read_const_val(tcx, DUMMY_SP, typing_env, CanAccessMutGlobal::No);
            let imm = valtree_to_ref(&mut ecx, cv.valtree, inner_ty);
            let imm = ImmTy::from_immediate(
                imm,
                tcx.layout_of(typing_env.as_query_input(cv.ty)).unwrap(),
            );
            op_to_const(&ecx, &imm.into(), /* for diagnostics */ false)
        }
        ty::Tuple(_) | ty::Array(_, _) | ty::Adt(..) => {
            let layout = tcx.layout_of(typing_env.as_query_input(cv.ty)).unwrap();
            if layout.is_zst() {
                // Fast path to avoid some allocations.
                return mir::ConstValue::ZeroSized;
            }
            if layout.backend_repr.is_scalar()
                && (matches!(cv.ty.kind(), ty::Tuple(_))
                    || matches!(cv.ty.kind(), ty::Adt(def, _) if def.is_struct()))
            {
                // A Scalar tuple/struct; we can avoid creating an allocation.
                let branches = cv.valtree.unwrap_branch();
                // Find the non-ZST field. (There can be aligned ZST!)
                for (i, &inner_valtree) in branches.iter().enumerate() {
                    let field = layout.field(&LayoutCx::new(tcx, typing_env), i);
                    if !field.is_zst() {
                        let cv = ty::Value { valtree: inner_valtree, ty: field.ty };
                        return valtree_to_const_value(tcx, typing_env, cv);
                    }
                }
                bug!("could not find non-ZST field during in {layout:#?}");
            }

            let mut ecx =
                mk_eval_cx_to_read_const_val(tcx, DUMMY_SP, typing_env, CanAccessMutGlobal::No);

            // Need to create a place for this valtree.
            let place = create_valtree_place(&mut ecx, layout, cv.valtree);

            valtree_into_mplace(&mut ecx, &place, cv.valtree);
            dump_place(&ecx, &place);
            intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &place).unwrap();

            op_to_const(&ecx, &place.into(), /* for diagnostics */ false)
        }
        ty::Never
        | ty::Error(_)
        | ty::Foreign(..)
        | ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        | ty::Infer(_)
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Coroutine(..)
        | ty::CoroutineWitness(..)
        | ty::FnPtr(..)
        | ty::Str
        | ty::Slice(_)
        | ty::Dynamic(..)
        | ty::UnsafeBinder(_) => {
            bug!("no ValTree should have been created for type {:?}", cv.ty.kind())
        }
    }
}

/// Put a valtree into memory and return a reference to that.
fn valtree_to_ref<'tcx>(
    ecx: &mut CompileTimeInterpCx<'tcx>,
    valtree: ty::ValTree<'tcx>,
    pointee_ty: Ty<'tcx>,
) -> Immediate {
    let pointee_place = create_valtree_place(ecx, ecx.layout_of(pointee_ty).unwrap(), valtree);
    debug!(?pointee_place);

    valtree_into_mplace(ecx, &pointee_place, valtree);
    dump_place(ecx, &pointee_place);
    intern_const_alloc_recursive(ecx, InternKind::Constant, &pointee_place).unwrap();

    pointee_place.to_ref(&ecx.tcx)
}

#[instrument(skip(ecx), level = "debug")]
fn valtree_into_mplace<'tcx>(
    ecx: &mut CompileTimeInterpCx<'tcx>,
    place: &MPlaceTy<'tcx>,
    valtree: ty::ValTree<'tcx>,
) {
    // This will match on valtree and write the value(s) corresponding to the ValTree
    // inside the place recursively.

    let ty = place.layout.ty;

    match ty.kind() {
        ty::FnDef(_, _) => {
            // Zero-sized type, nothing to do.
        }
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Char | ty::RawPtr(..) => {
            let scalar_int = valtree.unwrap_leaf();
            debug!("writing trivial valtree {:?} to place {:?}", scalar_int, place);
            ecx.write_immediate(Immediate::Scalar(scalar_int.into()), place).unwrap();
        }
        ty::Ref(_, inner_ty, _) => {
            let imm = valtree_to_ref(ecx, valtree, *inner_ty);
            debug!(?imm);
            ecx.write_immediate(imm, place).unwrap();
        }
        ty::Adt(_, _) | ty::Tuple(_) | ty::Array(_, _) | ty::Str | ty::Slice(_) => {
            let branches = valtree.unwrap_branch();

            // Need to downcast place for enums
            let (place_adjusted, branches, variant_idx) = match ty.kind() {
                ty::Adt(def, _) if def.is_enum() => {
                    // First element of valtree corresponds to variant
                    let scalar_int = branches[0].unwrap_leaf();
                    let variant_idx = VariantIdx::from_u32(scalar_int.to_u32());
                    let variant = def.variant(variant_idx);
                    debug!(?variant);

                    (
                        ecx.project_downcast(place, variant_idx).unwrap(),
                        &branches[1..],
                        Some(variant_idx),
                    )
                }
                _ => (place.clone(), branches, None),
            };
            debug!(?place_adjusted, ?branches);

            // Create the places (by indexing into `place`) for the fields and fill
            // them recursively
            for (i, inner_valtree) in branches.iter().enumerate() {
                debug!(?i, ?inner_valtree);

                let place_inner = match ty.kind() {
                    ty::Str | ty::Slice(_) | ty::Array(..) => {
                        ecx.project_index(place, i as u64).unwrap()
                    }
                    _ => ecx.project_field(&place_adjusted, i).unwrap(),
                };

                debug!(?place_inner);
                valtree_into_mplace(ecx, &place_inner, *inner_valtree);
                dump_place(ecx, &place_inner);
            }

            debug!("dump of place_adjusted:");
            dump_place(ecx, &place_adjusted);

            if let Some(variant_idx) = variant_idx {
                // don't forget filling the place with the discriminant of the enum
                ecx.write_discriminant(variant_idx, place).unwrap();
            }

            debug!("dump of place after writing discriminant:");
            dump_place(ecx, place);
        }
        _ => bug!("shouldn't have created a ValTree for {:?}", ty),
    }
}

fn dump_place<'tcx>(ecx: &CompileTimeInterpCx<'tcx>, place: &MPlaceTy<'tcx>) {
    trace!("{:?}", ecx.dump_place(&PlaceTy::from(place.clone())));
}
