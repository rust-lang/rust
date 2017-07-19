mod cast;
mod const_eval;
mod error;
mod eval_context;
mod lvalue;
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
    eval_main,
};

pub use self::lvalue::{
    Lvalue,
    LvalueExtra,
};

pub use self::memory::{
    AllocId,
    Memory,
    MemoryPointer,
};

pub use self::value::{
    PrimVal,
    PrimValKind,
    Value,
    Pointer,
};

pub use self::const_eval::{
    eval_body_as_integer,
};
