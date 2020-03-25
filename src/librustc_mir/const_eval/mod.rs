// Not in interpret to make sure we do not use private implementation details

use std::convert::TryFrom;

use rustc::mir;
use rustc::ty::layout::VariantIdx;
use rustc::ty::{self, TyCtxt};
use rustc_span::{source_map::DUMMY_SP, symbol::Symbol};

use crate::interpret::{intern_const_alloc_recursive, ConstValue, InternKind, InterpCx};

mod error;
mod eval_queries;
mod fn_queries;
mod machine;

pub use error::*;
pub use eval_queries::*;
pub use fn_queries::*;
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
) -> ConstValue<'tcx> {
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
    let field = ecx.operand_field(down, field.index()).unwrap();
    // and finally move back to the const world, always normalizing because
    // this is not called for statics.
    op_to_const(&ecx, field)
}

pub(crate) fn const_caller_location(
    tcx: TyCtxt<'tcx>,
    (file, line, col): (Symbol, u32, u32),
) -> ConstValue<'tcx> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx(tcx, DUMMY_SP, ty::ParamEnv::reveal_all(), false);

    let loc_place = ecx.alloc_caller_location(file, line, col);
    intern_const_alloc_recursive(&mut ecx, InternKind::Constant, loc_place, false).unwrap();
    ConstValue::Scalar(loc_place.ptr)
}

// this function uses `unwrap` copiously, because an already validated constant
// must have valid fields and can thus never fail outside of compiler bugs
pub(crate) fn destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> mir::DestructuredConst<'tcx> {
    trace!("destructure_const: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.eval_const_to_op(val, None).unwrap();

    let variant = ecx.read_discriminant(op).unwrap().1;

    // We go to `usize` as we cannot allocate anything bigger anyway.
    let field_count = match val.ty.kind {
        ty::Array(_, len) => usize::try_from(len.eval_usize(tcx, param_env)).unwrap(),
        ty::Adt(def, _) => def.variants[variant].fields.len(),
        ty::Tuple(substs) => substs.len(),
        _ => bug!("cannot destructure constant {:?}", val),
    };

    let down = ecx.operand_downcast(op, variant).unwrap();
    let fields_iter = (0..field_count).map(|i| {
        let field_op = ecx.operand_field(down, i).unwrap();
        let val = op_to_const(&ecx, field_op);
        ty::Const::from_value(tcx, val, field_op.layout.ty)
    });
    let fields = tcx.arena.alloc_from_iter(fields_iter);

    mir::DestructuredConst { variant, fields }
}
