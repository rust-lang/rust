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

pub use self::eval_context::{EvalContext, Frame, ResourceLimits, StackPopCleanup,
                             TyAndPacked, ValTy};

pub use self::place::{Place, PlaceExtra};

pub use self::memory::{Memory, MemoryKind, HasMemory};

pub use self::const_eval::{
    eval_body_with_mir,
    mk_borrowck_eval_cx,
    eval_body,
    CompileTimeEvaluator,
    const_eval_provider,
    const_val_field,
    const_discr,
};

pub use self::machine::Machine;

pub use self::operator::unary_op;
pub use self::memory::{write_target_uint, write_target_int, read_target_uint, read_target_int};
