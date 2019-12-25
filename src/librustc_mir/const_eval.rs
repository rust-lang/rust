// Not in interpret to make sure we do not use private implementation details

use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::hash::Hash;

use rustc::mir;
use rustc::ty::layout::{self, VariantIdx};
use rustc::ty::{self, TyCtxt};

use syntax::{
    source_map::{Span, DUMMY_SP},
    symbol::Symbol,
};

use crate::interpret::{
    intern_const_alloc_recursive, Allocation, ConstValue, ImmTy, Immediate, InterpCx, OpTy, Scalar,
};

mod error;
mod query;

pub use error::*;
pub use query::*;

/// The `InterpCx` is only meant to be used to do field and index projections into constants for
/// `simd_shuffle` and const patterns in match arms.
///
/// The function containing the `match` that is currently being analyzed may have generic bounds
/// that inform us about the generic bounds of the constant. E.g., using an associated constant
/// of a function's generic parameter will require knowledge about the bounds on the generic
/// parameter. These bounds are passed to `mk_eval_cx` via the `ParamEnv` argument.
fn mk_eval_cx<'mir, 'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
    can_access_statics: bool,
) -> CompileTimeEvalContext<'mir, 'tcx> {
    debug!("mk_eval_cx: {:?}", param_env);
    InterpCx::new(
        tcx.at(span),
        param_env,
        CompileTimeInterpreter::new(),
        MemoryExtra { can_access_statics },
    )
}

fn op_to_const<'tcx>(
    ecx: &CompileTimeEvalContext<'_, 'tcx>,
    op: OpTy<'tcx>,
) -> &'tcx ty::Const<'tcx> {
    // We do not have value optimizations for everything.
    // Only scalars and slices, since they are very common.
    // Note that further down we turn scalars of undefined bits back to `ByRef`. These can result
    // from scalar unions that are initialized with one of their zero sized variants. We could
    // instead allow `ConstValue::Scalar` to store `ScalarMaybeUndef`, but that would affect all
    // the usual cases of extracting e.g. a `usize`, without there being a real use case for the
    // `Undef` situation.
    let try_as_immediate = match op.layout.abi {
        layout::Abi::Scalar(..) => true,
        layout::Abi::ScalarPair(..) => match op.layout.ty.kind {
            ty::Ref(_, inner, _) => match inner.kind {
                ty::Slice(elem) => elem == ecx.tcx.types.u8,
                ty::Str => true,
                _ => false,
            },
            _ => false,
        },
        _ => false,
    };
    let immediate = if try_as_immediate {
        Err(ecx.read_immediate(op).expect("normalization works on validated constants"))
    } else {
        // It is guaranteed that any non-slice scalar pair is actually ByRef here.
        // When we come back from raw const eval, we are always by-ref. The only way our op here is
        // by-val is if we are in const_field, i.e., if this is (a field of) something that we
        // "tried to make immediate" before. We wouldn't do that for non-slice scalar pairs or
        // structs containing such.
        op.try_as_mplace()
    };
    let val = match immediate {
        Ok(mplace) => {
            let ptr = mplace.ptr.to_ptr().unwrap();
            let alloc = ecx.tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id);
            ConstValue::ByRef { alloc, offset: ptr.offset }
        }
        // see comment on `let try_as_immediate` above
        Err(ImmTy { imm: Immediate::Scalar(x), .. }) => match x {
            ScalarMaybeUndef::Scalar(s) => ConstValue::Scalar(s),
            ScalarMaybeUndef::Undef => {
                // When coming out of "normal CTFE", we'll always have an `Indirect` operand as
                // argument and we will not need this. The only way we can already have an
                // `Immediate` is when we are called from `const_field`, and that `Immediate`
                // comes from a constant so it can happen have `Undef`, because the indirect
                // memory that was read had undefined bytes.
                let mplace = op.assert_mem_place();
                let ptr = mplace.ptr.to_ptr().unwrap();
                let alloc = ecx.tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id);
                ConstValue::ByRef { alloc, offset: ptr.offset }
            }
        },
        Err(ImmTy { imm: Immediate::ScalarPair(a, b), .. }) => {
            let (data, start) = match a.not_undef().unwrap() {
                Scalar::Ptr(ptr) => {
                    (ecx.tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id), ptr.offset.bytes())
                }
                Scalar::Raw { .. } => (
                    ecx.tcx.intern_const_alloc(Allocation::from_byte_aligned_bytes(b"" as &[u8])),
                    0,
                ),
            };
            let len = b.to_machine_usize(&ecx.tcx.tcx).unwrap();
            let start = start.try_into().unwrap();
            let len: usize = len.try_into().unwrap();
            ConstValue::Slice { data, start, end: start + len }
        }
    };
    ecx.tcx.mk_const(ty::Const { val: ty::ConstKind::Value(val), ty: op.layout.ty })
}

/// Extracts a field of a (variant of a) const.
// this function uses `unwrap` copiously, because an already validated constant must have valid
// fields and can thus never fail outside of compiler bugs
pub fn const_field<'tcx>(
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

pub fn const_caller_location<'tcx>(
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
pub fn const_variant_index<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> VariantIdx {
    trace!("const_variant_index: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env, false);
    let op = ecx.eval_const_to_op(val, None).unwrap();
    ecx.read_discriminant(op).unwrap().1
}
