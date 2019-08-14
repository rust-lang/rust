#![feature(rustc_private)]

#![warn(rust_2018_idioms)]
#![allow(clippy::cast_lossless)]

#[macro_use]
extern crate log;
// From rustc.
extern crate syntax;
extern crate rustc_apfloat;
#[macro_use] extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate rustc_target;

mod shims;
mod operator;
mod helpers;
mod range_map;
mod mono_hash_map;
mod stacked_borrows;
mod intptrcast;
mod machine;
mod eval;

// Make all those symbols available in the same place as our own.
pub use rustc_mir::interpret::*;
// Resolve ambiguity.
pub use rustc_mir::interpret::{self, AllocMap, PlaceTy};

pub use crate::shims::{EvalContextExt as ShimsEvalContextExt};
pub use crate::shims::foreign_items::EvalContextExt as ForeignItemsEvalContextExt;
pub use crate::shims::intrinsics::EvalContextExt as IntrinsicsEvalContextExt;
pub use crate::shims::tls::{EvalContextExt as TlsEvalContextExt, TlsData};
pub use crate::shims::dlsym::{Dlsym, EvalContextExt as DlsymEvalContextExt};
pub use crate::shims::env::{EnvVars, EvalContextExt as EnvEvalContextExt};
pub use crate::operator::EvalContextExt as OperatorEvalContextExt;
pub use crate::range_map::RangeMap;
pub use crate::helpers::{EvalContextExt as HelpersEvalContextExt};
pub use crate::mono_hash_map::MonoHashMap;
pub use crate::stacked_borrows::{EvalContextExt as StackedBorEvalContextExt, Tag, Permission, Stack, Stacks, Item};
pub use crate::machine::{
    PAGE_SIZE, STACK_ADDR, STACK_SIZE, NUM_CPUS,
    MemoryExtra, AllocExtra, MiriMemoryKind, Evaluator, MiriEvalContext, MiriEvalContextExt,
};
pub use crate::eval::{eval_main, create_ecx, MiriConfig};

/// Insert rustc arguments at the beginning of the argument list that Miri wants to be
/// set per default, for maximal validation power.
pub fn miri_default_args() -> &'static [&'static str] {
    &["-Zalways-encode-mir", "-Zmir-emit-retag", "-Zmir-opt-level=0", "--cfg=miri"]
}
