#![feature(rustc_private)]

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

mod fn_call;
mod operator;
mod intrinsic;
mod helpers;
mod tls;
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

pub use crate::fn_call::EvalContextExt as MissingFnsEvalContextExt;
pub use crate::operator::EvalContextExt as OperatorEvalContextExt;
pub use crate::intrinsic::EvalContextExt as IntrinsicEvalContextExt;
pub use crate::tls::{EvalContextExt as TlsEvalContextExt, TlsData};
pub use crate::range_map::RangeMap;
pub use crate::helpers::{EvalContextExt as HelpersEvalContextExt};
pub use crate::mono_hash_map::MonoHashMap;
pub use crate::stacked_borrows::{EvalContextExt as StackedBorEvalContextExt, Tag, Permission, Stack, Stacks, Item};
pub use crate::machine::{MemoryExtra, AllocExtra, MiriMemoryKind, Evaluator, MiriEvalContext, MiriEvalContextExt};
pub use crate::eval::{eval_main, create_ecx, MiriConfig};

// Some global facts about the emulated machine.
pub const PAGE_SIZE: u64 = 4*1024;
pub const STACK_ADDR: u64 = 16*PAGE_SIZE; // not really about the "stack", but where we start assigning integer addresses to allocations
pub const NUM_CPUS: u64 = 1;

/// Insert rustc arguments at the beginning of the argument list that Miri wants to be
/// set per default, for maximal validation power.
pub fn miri_default_args() -> &'static [&'static str] {
    // The flags here should be kept in sync with what bootstrap adds when `test-miri` is
    // set, which happens in `bootstrap/bin/rustc.rs` in the rustc sources.
    &["-Zalways-encode-mir", "-Zmir-emit-retag", "-Zmir-opt-level=0", "--cfg=miri"]
}
