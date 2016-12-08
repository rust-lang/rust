#![feature(
    btree_range,
    cell_extras,
    collections,
    collections_bound,
    pub_restricted,
    rustc_private,
)]

// From rustc.
#[macro_use]
extern crate log;
extern crate log_settings;
#[macro_use]
extern crate rustc;
extern crate rustc_borrowck;
extern crate rustc_const_math;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate syntax;

// From crates.io.
extern crate byteorder;

mod cast;
mod error;
mod eval_context;
mod lvalue;
mod memory;
mod primval;
mod step;
mod terminator;
mod value;
mod vtable;

pub use error::{
    EvalError,
    EvalResult,
};

pub use eval_context::{
    EvalContext,
    Frame,
    ResourceLimits,
    StackPopCleanup,
    Value,
    eval_main,
    run_mir_passes,
};

pub use lvalue::{
    Lvalue,
    LvalueExtra,
};

pub use memory::{
    Memory,
    Pointer,
    AllocId,
};

pub use primval::{
    PrimVal,
    PrimValKind,
};
