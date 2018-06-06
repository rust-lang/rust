//! An interpreter for MIR used in CTFE and by miri

mod cast;
mod const_eval;
mod eval_context;
mod place;
mod machine;
mod memory;
mod operator;
mod step;
mod terminator;
mod traits;

pub use self::eval_context::{EvalContext, Frame, StackPopCleanup,
                             TyAndPacked, ValTy};

pub use self::place::{Place, PlaceExtra};

pub use self::memory::{Memory, MemoryKind, HasMemory};

pub use self::const_eval::{
    eval_promoted,
    mk_borrowck_eval_cx,
    CompileTimeEvaluator,
    const_value_to_allocation_provider,
    const_eval_provider,
    const_val_field,
    const_variant_index,
    value_to_const_value,
};

pub use self::machine::Machine;

pub use self::memory::{write_target_uint, write_target_int, read_target_uint};

use rustc::mir::interpret::{EvalResult, EvalErrorKind};
use rustc::ty::{Ty, TyCtxt, ParamEnv};

pub fn sign_extend<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, value: u128, ty: Ty<'tcx>) -> EvalResult<'tcx, u128> {
    let param_env = ParamEnv::empty();
    let layout = tcx.layout_of(param_env.and(ty)).map_err(|layout| EvalErrorKind::Layout(layout))?;
    let size = layout.size.bits();
    assert!(layout.abi.is_signed());
    // sign extend
    let shift = 128 - size;
    // shift the unsigned value to the left
    // and back to the right as signed (essentially fills with FF on the left)
    Ok((((value << shift) as i128) >> shift) as u128)
}

pub fn truncate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, value: u128, ty: Ty<'tcx>) -> EvalResult<'tcx, u128> {
    let param_env = ParamEnv::empty();
    let layout = tcx.layout_of(param_env.and(ty)).map_err(|layout| EvalErrorKind::Layout(layout))?;
    let size = layout.size.bits();
    let shift = 128 - size;
    // truncate (shift left to drop out leftover values, shift right to fill with zeroes)
    Ok((value << shift) >> shift)
}
