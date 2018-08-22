//! An interpreter for MIR used in CTFE and by miri

mod cast;
mod eval_context;
mod place;
mod operand;
mod machine;
mod memory;
mod operator;
mod step;
mod terminator;
mod traits;
mod const_eval;
mod validity;

pub use self::eval_context::{
    EvalContext, Frame, StackPopCleanup, LocalValue,
};

pub use self::place::{Place, PlaceExtra, PlaceTy, MemPlace, MPlaceTy};

pub use self::memory::{Memory, MemoryKind, HasMemory};

pub use self::const_eval::{
    eval_promoted,
    mk_borrowck_eval_cx,
    mk_eval_cx,
    CompileTimeEvaluator,
    const_to_allocation_provider,
    const_eval_provider,
    const_field,
    const_variant_index,
    op_to_const,
};

pub use self::machine::Machine;

pub use self::operand::{Value, ValTy, Operand, OpTy};
