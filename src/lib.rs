#![feature(rustc_private)]
#![feature(map_first_last)]
#![feature(map_try_insert)]
#![feature(never_type)]
#![feature(try_blocks)]
#![feature(let_else)]
#![feature(bool_to_option)]
#![feature(io_error_more)]
#![warn(rust_2018_idioms)]
#![allow(clippy::cast_lossless)]

extern crate rustc_apfloat;
extern crate rustc_ast;
#[macro_use]
extern crate rustc_middle;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

mod data_race;
mod diagnostics;
mod eval;
mod helpers;
mod intptrcast;
mod machine;
mod mono_hash_map;
mod operator;
mod range_map;
mod shims;
mod stacked_borrows;
mod sync;
mod thread;
mod vector_clock;

// Establish a "crate-wide prelude": we often import `crate::*`.

// Make all those symbols available in the same place as our own.
pub use rustc_const_eval::interpret::*;
// Resolve ambiguity.
pub use rustc_const_eval::interpret::{self, AllocMap, PlaceTy};

pub use crate::shims::dlsym::{Dlsym, EvalContextExt as _};
pub use crate::shims::env::{EnvVars, EvalContextExt as _};
pub use crate::shims::foreign_items::EvalContextExt as _;
pub use crate::shims::intrinsics::EvalContextExt as _;
pub use crate::shims::os_str::EvalContextExt as _;
pub use crate::shims::panic::{CatchUnwindData, EvalContextExt as _};
pub use crate::shims::time::EvalContextExt as _;
pub use crate::shims::tls::{EvalContextExt as _, TlsData};
pub use crate::shims::EvalContextExt as _;

pub use crate::data_race::{
    AtomicFenceOp, AtomicReadOp, AtomicRwOp, AtomicWriteOp,
    EvalContextExt as DataRaceEvalContextExt,
};
pub use crate::diagnostics::{
    register_diagnostic, report_error, EvalContextExt as DiagnosticsEvalContextExt,
    NonHaltingDiagnostic, TerminationInfo,
};
pub use crate::eval::{
    create_ecx, eval_entry, AlignmentCheck, BacktraceStyle, IsolatedOp, MiriConfig, RejectOpWith,
};
pub use crate::helpers::EvalContextExt as HelpersEvalContextExt;
pub use crate::machine::{
    AllocExtra, Evaluator, FrameData, MemoryExtra, MiriEvalContext, MiriEvalContextExt,
    MiriMemoryKind, Tag, NUM_CPUS, PAGE_SIZE, STACK_ADDR, STACK_SIZE,
};
pub use crate::mono_hash_map::MonoHashMap;
pub use crate::operator::EvalContextExt as OperatorEvalContextExt;
pub use crate::range_map::RangeMap;
pub use crate::stacked_borrows::{
    CallId, EvalContextExt as StackedBorEvalContextExt, Item, Permission, PtrId, SbTag, Stack,
    Stacks,
};
pub use crate::sync::{CondvarId, EvalContextExt as SyncEvalContextExt, MutexId, RwLockId};
pub use crate::thread::{
    EvalContextExt as ThreadsEvalContextExt, SchedulingAction, ThreadId, ThreadManager, ThreadState,
};
pub use crate::vector_clock::{VClock, VTimestamp, VectorIdx};

/// Insert rustc arguments at the beginning of the argument list that Miri wants to be
/// set per default, for maximal validation power.
pub const MIRI_DEFAULT_ARGS: &[&str] = &[
    "-Zalways-encode-mir",
    "-Zmir-emit-retag",
    "-Zmir-opt-level=0",
    "--cfg=miri",
    "-Cdebug-assertions=on",
];
