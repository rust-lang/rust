#![feature(
    btree_range,
    collections,
    collections_bound,
    question_mark,
    rustc_private,
    pub_restricted,
)]

// From rustc.
#[macro_use] extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate rustc_trans;
extern crate rustc_const_math;
extern crate syntax;
#[macro_use] extern crate log;
extern crate log_settings;

// From crates.io.
extern crate byteorder;

mod error;
mod interpreter;
mod memory;
mod primval;

pub use error::{
    EvalError,
    EvalResult,
};

pub use interpreter::{
    CachedMir,
    EvalContext,
    Frame,
    eval_main,
};

pub use memory::{
    Memory,
    Pointer,
    AllocId,
};
