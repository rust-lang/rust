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

pub use self::eval_context::{
    EvalContext, Frame, StackPopCleanup,
    TyAndPacked, ValTy,
};

pub use self::place::{Place, PlaceExtra};

pub use self::memory::{Memory, MemoryKind, HasMemory};

pub use self::const_eval::{
    eval_promoted,
    mk_borrowck_eval_cx,
    mk_eval_cx,
    CompileTimeEvaluator,
    const_value_to_allocation_provider,
    const_eval_provider,
    const_val_field,
    const_variant_index,
    value_to_const_value,
};

pub use self::machine::Machine;

pub use self::memory::{write_target_uint, write_target_int, read_target_uint};

use rustc::ty::layout::TyLayout;

pub fn sign_extend(value: u128, layout: TyLayout<'_>) -> u128 {
    let size = layout.size.bits();
    assert!(layout.abi.is_signed());
    // sign extend
    let shift = 128 - size;
    // shift the unsigned value to the left
    // and back to the right as signed (essentially fills with FF on the left)
    (((value << shift) as i128) >> shift) as u128
}

pub fn truncate(value: u128, layout: TyLayout<'_>) -> u128 {
    let size = layout.size.bits();
    let shift = 128 - size;
    // truncate (shift left to drop out leftover values, shift right to fill with zeroes)
    (value << shift) >> shift
}
