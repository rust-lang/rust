#![feature(
    btree_range,
    collections,
    collections_bound,
    core_intrinsics,
    filling_drop,
    question_mark,
    rustc_private,
    pub_restricted,
)]

// From rustc.
#[macro_use] extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate rustc_trans;
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
    step,
};

pub use memory::Memory;
