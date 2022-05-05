// Not in interpret to make sure we do not use private implementation details

use std::convert::TryFrom;

use rustc_hir::Mutability;
use rustc_middle::mir;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{source_map::DUMMY_SP, symbol::Symbol};

use crate::interpret::{
    intern_const_alloc_recursive, ConstValue, InternKind, InterpCx, InterpResult, MemPlaceMeta,
    Scalar,
};

mod error;
mod eval_queries;
mod fn_queries;
mod machine;
mod valtrees;

pub use error::*;
pub use eval_queries::*;
pub use fn_queries::*;
pub use machine::*;
pub(crate) use valtrees::{const_to_valtree, valtree_to_const_value};

pub(crate) fn const_caller_location(
    tcx: TyCtxt<'_>,
    (file, line, col): (Symbol, u32, u32),
) -> ConstValue<'_> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx(tcx, DUMMY_SP, ty::ParamEnv::reveal_all(), false);

    let loc_place = ecx.alloc_caller_location(file, line, col);
    if intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &loc_place).is_err() {
        bug!("intern_const_alloc_recursive should not error in this case")
    }
    ConstValue::Scalar(Scalar::from_maybe_pointer(loc_place.ptr, &tcx))
}

/// This function should never fail for validated constants. However, it is also invoked from the
/// pretty printer which might attempt to format invalid constants and in that case it might fail.
pub(crate) fn try_destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: ty::Const<'tcx>,
) -> InterpResult<'tcx, mir::DestructuredConst<'tcx>> {
    trace!("destructure_const: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.const_to_op(val, None)?;

    // We go to `usize` as we cannot allocate anything bigger anyway.
    let (field_count, variant, down) = match val.ty().kind() {
        ty::Array(_, len) => (usize::try_from(len.eval_usize(tcx, param_env)).unwrap(), None, op),
        // Checks if we have any variants, to avoid downcasting to a non-existing variant (when
        // there are no variants `read_discriminant` successfully returns a non-existing variant
        // index).
        ty::Adt(def, _) if def.variants().is_empty() => throw_ub!(Unreachable),
        ty::Adt(def, _) => {
            let variant = ecx.read_discriminant(&op)?.1;
            let down = ecx.operand_downcast(&op, variant)?;
            (def.variant(variant).fields.len(), Some(variant), down)
        }
        ty::Tuple(substs) => (substs.len(), None, op),
        _ => bug!("cannot destructure constant {:?}", val),
    };

    let fields = (0..field_count)
        .map(|i| {
            let field_op = ecx.operand_field(&down, i)?;
            let val = op_to_const(&ecx, &field_op);
            Ok(ty::Const::from_value(tcx, val, field_op.layout.ty))
        })
        .collect::<InterpResult<'tcx, Vec<_>>>()?;
    let fields = tcx.arena.alloc_from_iter(fields);

    Ok(mir::DestructuredConst { variant, fields })
}

#[instrument(skip(tcx), level = "debug")]
pub(crate) fn deref_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: ty::Const<'tcx>,
) -> ty::Const<'tcx> {
    trace!("deref_const: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.const_to_op(val, None).unwrap();
    let mplace = ecx.deref_operand(&op).unwrap();
    if let Some(alloc_id) = mplace.ptr.provenance {
        assert_eq!(
            tcx.get_global_alloc(alloc_id).unwrap().unwrap_memory().inner().mutability,
            Mutability::Not,
            "deref_const cannot be used with mutable allocations as \
            that could allow pattern matching to observe mutable statics",
        );
    }

    let ty = match mplace.meta {
        MemPlaceMeta::None => mplace.layout.ty,
        MemPlaceMeta::Poison => bug!("poison metadata in `deref_const`: {:#?}", mplace),
        // In case of unsized types, figure out the real type behind.
        MemPlaceMeta::Meta(scalar) => match mplace.layout.ty.kind() {
            ty::Str => bug!("there's no sized equivalent of a `str`"),
            ty::Slice(elem_ty) => tcx.mk_array(*elem_ty, scalar.to_machine_usize(&tcx).unwrap()),
            _ => bug!(
                "type {} should not have metadata, but had {:?}",
                mplace.layout.ty,
                mplace.meta
            ),
        },
    };

    tcx.mk_const(ty::ConstS { val: ty::ConstKind::Value(op_to_const(&ecx, &mplace.into())), ty })
}
