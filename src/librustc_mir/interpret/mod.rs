//! An interpreter for MIR used in CTFE and by miri

mod cast;
mod const_eval;
mod error;
mod eval_context;
mod lvalue;
mod validation;
mod machine;
mod memory;
mod operator;
mod step;
mod terminator;
mod traits;
mod value;

pub use self::error::{
    EvalError,
    EvalResult,
};

pub use self::eval_context::{
    EvalContext,
    Frame,
    ResourceLimits,
    StackPopCleanup,
    DynamicLifetime,
    TyAndPacked,
};

pub use self::lvalue::{
    Lvalue,
    LvalueExtra,
    Global,
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
    LockInfo,
    AccessKind,
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
