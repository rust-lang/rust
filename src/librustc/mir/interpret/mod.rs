//! An interpreter for MIR used in CTFE and by miri

#[macro_export]
macro_rules! err {
    ($($tt:tt)*) => { Err($crate::mir::interpret::EvalErrorKind::$($tt)*.into()) };
}

mod cast;
mod const_eval;
mod error;
mod eval_context;
mod place;
mod validation;
mod machine;
mod memory;
mod operator;
mod range_map;
mod step;
mod terminator;
mod traits;
mod value;

pub use self::error::{EvalError, EvalResult, EvalErrorKind};

pub use self::eval_context::{EvalContext, Frame, ResourceLimits, StackPopCleanup, DynamicLifetime,
                             TyAndPacked, PtrAndAlign, ValTy};

pub use self::place::{Place, PlaceExtra, GlobalId};

pub use self::memory::{AllocId, Memory, MemoryPointer, MemoryKind, HasMemory, AccessKind, Allocation};

use self::memory::{PointerArithmetic, Lock};

use self::range_map::RangeMap;

pub use self::value::{PrimVal, PrimValKind, Value, Pointer};

pub use self::const_eval::{eval_body_as_integer, eval_body, CompileTimeEvaluator};

pub use self::machine::Machine;

pub use self::validation::{ValidationQuery, AbsPlace};
