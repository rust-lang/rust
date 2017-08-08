//! An interpreter for MIR used in CTFE and by miri

#[macro_export]
macro_rules! err {
    ($($tt:tt)*) => { Err($crate::interpret::EvalErrorKind::$($tt)*.into()) };
}

mod cast;
mod const_eval;
mod error;
mod eval_context;
mod lvalue;
mod validation;
mod machine;
mod memory;
mod operator;
mod range_map;
mod step;
mod terminator;
mod traits;
mod value;

pub use self::error::{
    EvalError,
    EvalResult,
    EvalErrorKind,
};

pub use self::eval_context::{
    EvalContext,
    Frame,
    ResourceLimits,
    StackPopCleanup,
    DynamicLifetime,
    TyAndPacked,
    PtrAndAlign,
};

pub use self::lvalue::{
    Lvalue,
    LvalueExtra,
    GlobalId,
};

pub use self::memory::{
    AllocId,
    Memory,
    MemoryPointer,
    Kind,
    HasMemory,
};

use self::memory::{
    PointerArithmetic,
    Lock,
    AccessKind,
};

use self::range_map::{
    RangeMap
};

pub use self::value::{
    PrimVal,
    PrimValKind,
    Value,
    Pointer,
};

pub use self::const_eval::{
    eval_body_as_integer,
    eval_body_as_primval,
};

pub use self::machine::{
    Machine,
};

pub use self::validation::{
    ValidationQuery,
};
