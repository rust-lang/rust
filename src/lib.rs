#![feature(rustc_private)]
#![feature(option_expect_none, option_unwrap_none)]
#![warn(rust_2018_idioms)]
#![allow(clippy::cast_lossless)]

#[macro_use]
extern crate log;
// From rustc.
extern crate rustc_apfloat;
extern crate syntax;
#[macro_use]
extern crate rustc;
extern crate rustc_hir;
extern crate rustc_span;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate rustc_target;

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

// Make all those symbols available in the same place as our own.
pub use rustc_mir::interpret::*;
// Resolve ambiguity.
pub use rustc_mir::interpret::{self, AllocMap, PlaceTy};

pub use crate::shims::dlsym::{Dlsym, EvalContextExt as DlsymEvalContextExt};
pub use crate::shims::env::{EnvVars, EvalContextExt as EnvEvalContextExt};
pub use crate::shims::foreign_items::EvalContextExt as ForeignItemsEvalContextExt;
pub use crate::shims::fs::{EvalContextExt as FileEvalContextExt, FileHandler};
pub use crate::shims::intrinsics::EvalContextExt as IntrinsicsEvalContextExt;
pub use crate::shims::panic::{CatchUnwindData, EvalContextExt as PanicEvalContextExt};
pub use crate::shims::time::EvalContextExt as TimeEvalContextExt;
pub use crate::shims::tls::{EvalContextExt as TlsEvalContextExt, TlsData};
pub use crate::shims::EvalContextExt as ShimsEvalContextExt;

pub use crate::diagnostics::{
    register_diagnostic, report_diagnostic, EvalContextExt as DiagnosticsEvalContextExt, NonHaltingDiagnostic,
};
pub use crate::eval::{create_ecx, eval_main, MiriConfig, TerminationInfo};
pub use crate::helpers::EvalContextExt as HelpersEvalContextExt;
pub use crate::machine::{
    AllocExtra, Evaluator, FrameData, MemoryExtra, MiriEvalContext, MiriEvalContextExt,
    MiriMemoryKind, NUM_CPUS, PAGE_SIZE, STACK_ADDR, STACK_SIZE,
};
pub use crate::mono_hash_map::MonoHashMap;
pub use crate::operator::EvalContextExt as OperatorEvalContextExt;
pub use crate::range_map::RangeMap;
pub use crate::stacked_borrows::{
    EvalContextExt as StackedBorEvalContextExt, GlobalState, Item, Permission, PtrId, Stack,
    Stacks, Tag,
};

/// Insert rustc arguments at the beginning of the argument list that Miri wants to be
/// set per default, for maximal validation power.
pub fn miri_default_args() -> &'static [&'static str] {
    &["-Zalways-encode-mir", "-Zmir-emit-retag", "-Zmir-opt-level=0", "--cfg=miri"]
}
