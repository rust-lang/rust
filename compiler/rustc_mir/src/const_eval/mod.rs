// Not in interpret to make sure we do not use private implementation details

use std::convert::TryFrom;

use rustc_hir::Mutability;
use rustc_middle::mir;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{source_map::DUMMY_SP, symbol::Symbol};

use crate::interpret::{
    intern_const_alloc_recursive, ConstValue, InternKind, InterpCx, MemPlaceMeta, Scalar,
};

mod error;
mod eval_queries;
mod fn_queries;
mod machine;

pub use error::*;
pub use eval_queries::*;
pub use fn_queries::*;
pub use machine::*;

pub(crate) fn const_caller_location(
    tcx: TyCtxt<'tcx>,
    (file, line, col): (Symbol, u32, u32),
) -> ConstValue<'tcx> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx(tcx, DUMMY_SP, ty::ParamEnv::reveal_all(), false);

    let loc_place = ecx.alloc_caller_location(file, line, col);
    intern_const_alloc_recursive(&mut ecx, InternKind::Constant, loc_place);
    ConstValue::Scalar(loc_place.ptr)
}

/// This function uses `unwrap` copiously, because an already validated constant
/// must have valid fields and can thus never fail outside of compiler bugs. However, it is
/// invoked from the pretty printer, where it can receive enums with no variants and e.g.
/// `read_discriminant` needs to be able to handle that.
pub(crate) fn destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> mir::DestructuredConst<'tcx> {
    trace!("destructure_const: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.const_to_op(val, None).unwrap();

    // We go to `usize` as we cannot allocate anything bigger anyway.
    let (field_count, variant, down) = match val.ty.kind() {
        ty::Array(_, len) => (usize::try_from(len.eval_usize(tcx, param_env)).unwrap(), None, op),
        ty::Adt(def, _) if def.variants.is_empty() => {
            return mir::DestructuredConst { variant: None, fields: &[] };
        }
        ty::Adt(def, _) => {
            let variant = ecx.read_discriminant(op).unwrap().1;
            let down = ecx.operand_downcast(op, variant).unwrap();
            (def.variants[variant].fields.len(), Some(variant), down)
        }
        ty::Tuple(substs) => (substs.len(), None, op),
        _ => bug!("cannot destructure constant {:?}", val),
    };

    let fields_iter = (0..field_count).map(|i| {
        let field_op = ecx.operand_field(down, i).unwrap();
        let val = op_to_const(&ecx, field_op);
        ty::Const::from_value(tcx, val, field_op.layout.ty)
    });
    let fields = tcx.arena.alloc_from_iter(fields_iter);

    mir::DestructuredConst { variant, fields }
}

pub(crate) fn deref_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> &'tcx ty::Const<'tcx> {
    trace!("deref_const: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.const_to_op(val, None).unwrap();
    let mplace = ecx.deref_operand(op).unwrap();
    if let Scalar::Ptr(ptr) = mplace.ptr {
        assert_eq!(
            ecx.memory.get_raw(ptr.alloc_id).unwrap().mutability,
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
            ty::Slice(elem_ty) => tcx.mk_array(elem_ty, scalar.to_machine_usize(&tcx).unwrap()),
            _ => bug!(
                "type {} should not have metadata, but had {:?}",
                mplace.layout.ty,
                mplace.meta
            ),
        },
    };

    tcx.mk_const(ty::Const { val: ty::ConstKind::Value(op_to_const(&ecx, mplace.into())), ty })
}
