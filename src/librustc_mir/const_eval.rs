// Not in interpret to make sure we do not use private implementation details

use rustc::mir;
use rustc::ty::layout::VariantIdx;
use rustc::ty::{self, TyCtxt};
use rustc_span::{source_map::DUMMY_SP, symbol::Symbol};

use crate::interpret::{intern_const_alloc_recursive, ConstValue, InterpCx};

mod error;
mod eval_queries;
mod machine;

pub use error::*;
pub use eval_queries::*;
pub use machine::*;

/// Extracts a field of a (variant of a) const.
// this function uses `unwrap` copiously, because an already validated constant must have valid
// fields and can thus never fail outside of compiler bugs
pub(crate) fn const_field<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    variant: Option<VariantIdx>,
    field: mir::Field,
    value: &'tcx ty::Const<'tcx>,
) -> &'tcx ty::Const<'tcx> {
    trace!("const_field: {:?}, {:?}", field, value);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    // get the operand again
    let op = ecx.eval_const_to_op(value, None).unwrap();
    // downcast
    let down = match variant {
        None => op,
        Some(variant) => ecx.operand_downcast(op, variant).unwrap(),
    };
    // then project
    let field = ecx.operand_field(down, field.index() as u64).unwrap();
    // and finally move back to the const world, always normalizing because
    // this is not called for statics.
    op_to_const(&ecx, field)
}

pub(crate) fn const_caller_location<'tcx>(
    tcx: TyCtxt<'tcx>,
    (file, line, col): (Symbol, u32, u32),
) -> &'tcx ty::Const<'tcx> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx(tcx, DUMMY_SP, ty::ParamEnv::reveal_all(), false);

    let loc_ty = tcx.caller_location_ty();
    let loc_place = ecx.alloc_caller_location(file, line, col);
    intern_const_alloc_recursive(&mut ecx, None, loc_place).unwrap();
    let loc_const = ty::Const {
        ty: loc_ty,
        val: ty::ConstKind::Value(ConstValue::Scalar(loc_place.ptr.into())),
    };

    tcx.mk_const(loc_const)
}

// this function uses `unwrap` copiously, because an already validated constant must have valid
// fields and can thus never fail outside of compiler bugs
pub(crate) fn const_variant_index<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> VariantIdx {
    trace!("const_variant_index: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.eval_const_to_op(val, None).unwrap();
    ecx.read_discriminant(op).unwrap().1
}
